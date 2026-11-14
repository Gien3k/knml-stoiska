"""
Gra "Złudzenie sieci": sieć MLP (196 -> 96 -> 48 -> 10) rozpoznająca cyfry z kropek 14x14,
wytrenowana na prawdziwych odręcznych cyfrach MNIST (zmniejszonych 28x28 -> 14x14 i zbinaryzowanych).

Następnie automatycznie wyszukujemy poziomy: cyfra z MNIST + cyfra docelowa, dla których istnieje
zestaw 1-3 przełączonych kropek podnoszący pewność sieci dla celu powyżej 50%. Poziomy dzielimy
na 5 stopni trudności (1 kliknięcie wystarcza ... jedyne rozwiązanie z 3 kliknięć).
Wynik: frontend/model/digits-mlp.json
"""
import base64
import gzip
import itertools
import json
import os
import random
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "mnist")
OUT = os.path.join(HERE, "..", "frontend", "model", "digits-mlp.json")
MIRRORS = ["https://storage.googleapis.com/cvdf-datasets/mnist/", "https://ossci-datasets.s3.amazonaws.com/mnist/"]
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
G = 14
# pary (cyfra -> cel), które ludzie intuicyjnie "widzą" jako bliskie
PAIRS = {3: [8, 9, 5], 1: [7, 4], 5: [6, 9, 3], 9: [8, 4, 7], 4: [9], 6: [8, 5, 0], 0: [8, 6], 7: [9, 1, 2], 2: [3, 7], 8: [3, 9, 0]}
NAMES = ["zero", "jedynka", "dwójka", "trójka", "czwórka", "piątka", "szóstka", "siódemka", "ósemka", "dziewiątka"]
ACC = ["zero", "jedynkę", "dwójkę", "trójkę", "czwórkę", "piątkę", "szóstkę", "siódemkę", "ósemkę", "dziewiątkę"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fetch():
    os.makedirs(DATA, exist_ok=True)
    for f in FILES:
        p = os.path.join(DATA, f)
        if os.path.exists(p):
            continue
        for m in MIRRORS:
            try:
                urllib.request.urlretrieve(m + f, p)
                break
            except Exception as e:  # noqa
                log("mirror fail", m, e)
        else:
            raise RuntimeError("MNIST download failed")


def idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[2:3], "big")
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def to_dots(imgs):
    a = imgs.astype(np.float32) / 255.0
    a = a.reshape(-1, G, 2, G, 2).mean(axis=(2, 4))
    return (a > 0.45).astype(np.float32).reshape(-1, G * G)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(G * G, 96)
        self.l2 = nn.Linear(96, 48)
        self.l3 = nn.Linear(48, 10)

    def forward(self, x):
        return self.l3(F.relu(self.l2(F.relu(self.l1(x)))))


def shift(x):
    """losowe przesunięcie cyfry o -1..1 kropkę (augmentacja)"""
    b = x.view(-1, G, G)
    dx, dy = random.randint(-1, 1), random.randint(-1, 1)
    return torch.roll(b, shifts=(dy, dx), dims=(1, 2)).reshape(-1, G * G)


def main():
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    fetch()
    Xtr, Ytr = to_dots(idx(os.path.join(DATA, FILES[0]))), idx(os.path.join(DATA, FILES[1])).astype(np.int64)
    Xte, Yte = to_dots(idx(os.path.join(DATA, FILES[2]))), idx(os.path.join(DATA, FILES[3])).astype(np.int64)
    Xtr, Ytr, Xte, Yte = map(torch.from_numpy, (Xtr, Ytr, Xte, Yte))
    log("MNIST", Xtr.shape, Xte.shape)
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for ep in range(12):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            j = perm[i:i + 128]
            loss = F.cross_entropy(model(shift(Xtr[j])), Ytr[j])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xte).argmax(1) == Yte).float().mean().item()
        log(f"epoka {ep + 1} test_acc={acc:.4f}")

    # ------------------------------------------------------------ wyszukiwanie poziomów
    with torch.no_grad():
        P = F.softmax(model(Xte), 1)
    levels = []
    rng = random.Random(1)
    order = list(range(len(Xte)))
    rng.shuffle(order)
    per_pair = {}
    for n in order:
        y = int(Yte[n])
        if int(P[n].argmax()) != y or float(P[n, y]) < 0.95:
            continue
        x = Xte[n].clone()
        if x.sum() < 16 or x.sum() > 40:
            continue
        for t in PAIRS[y]:
            if per_pair.get((y, t), 0) >= 6:
                continue
            xg = x.clone().requires_grad_(True)
            lp = F.log_softmax(model(xg), 0)[t]
            g, = torch.autograd.grad(lp, xg)
            gain = (g * (1 - 2 * x)).numpy()
            cand = [int(i) for i in np.argsort(-gain)[:12]]
            best = None
            for k in (1, 2, 3):
                combos = list(itertools.combinations(cand, k))
                B = x.repeat(len(combos), 1)
                for r, c in enumerate(combos):
                    for i in c:
                        B[r, i] = 1 - B[r, i]
                with torch.no_grad():
                    pt = F.softmax(model(B), 1)[:, t]
                ok = [(combos[r], float(pt[r])) for r in range(len(combos)) if float(pt[r]) > 0.5]
                if ok:
                    best = (k, ok)
                    break
            if not best:
                continue
            k, ok = best
            # wabiki: kropki w pobliżu cyfry (żeby było w co klikać), razem 16 klikalnych miejsc
            grid = x.view(G, G).numpy()
            near = set()
            for r in range(G):
                for c in range(G):
                    if grid[max(0, r - 1):r + 2, max(0, c - 1):c + 2].sum() > 0:
                        near.add(r * G + c)
            decoys = [i for i in near if i not in cand]
            rng.shuffle(decoys)
            clickable = sorted(set(cand) | set(decoys[:16 - len(cand)]))
            levels.append({
                "src": y, "tgt": t, "minK": k, "nSol": len(ok),
                "x": "".join(str(int(v)) for v in x.tolist()),
                "click": clickable,
                "sol": [list(c) for c, _ in sorted(ok, key=lambda z: -z[1])[:3]],
                "p0": round(float(P[n, y]), 4),
            })
            per_pair[(y, t)] = per_pair.get((y, t), 0) + 1
        if len(levels) >= 400:
            break

    def tier(l):
        if l["minK"] == 1:
            return 1
        if l["minK"] == 2:
            return 2
        if l["nSol"] >= 4:
            return 3
        if l["nSol"] >= 2:
            return 4
        return 5
    for l in levels:
        l["tier"] = tier(l)
    pool = {}
    for l in levels:
        pool.setdefault(l["tier"], []).append(l)
    for k in sorted(pool):
        log(f"stopień {k}: {len(pool[k])} poziomów, pary: {sorted({(l['src'], l['tgt']) for l in pool[k]})}")

    b64 = lambda t: base64.b64encode(np.ascontiguousarray(t.detach().numpy().astype('<f4')).tobytes()).decode()
    out = {
        "name": "DigitMLP", "grid": G, "testAccuracy": round(acc, 4),
        "names": NAMES, "acc": ACC,
        "layers": [{"in": l.in_features, "out": l.out_features, "w": b64(l.weight), "b": b64(l.bias)} for l in (model.l1, model.l2, model.l3)],
        "levels": [l for k in sorted(pool) for l in pool[k][:30]],
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    # wektory testowe JS
    with torch.no_grad():
        tv = [{"x": "".join(str(int(v)) for v in Xte[i].tolist()), "p": [round(float(v), 6) for v in F.softmax(model(Xte[i]), 0)]} for i in range(5)]
    with open(os.path.join(HERE, "digits_test.json"), "w") as f:
        json.dump(tv, f)
    log("zapisano", OUT, os.path.getsize(OUT), "B; GOTOWE")


if __name__ == "__main__":
    main()
