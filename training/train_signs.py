"""
Gra "Zhakuj sieć neuronową": klasyfikator polskich znaków drogowych (CNN, wejście 48x48 RGB).

Zbiór treningowy jest syntetyczny: znaki rysowane programowo (kształt, kolory, napisy jak w
rozporządzeniu o znakach drogowych) na losowych tłach, z augmentacją: obrót, skala, przesunięcie,
rozmycie, szum, zmiana jasności i barw, przysłonięcia. Wynik:
  frontend/model/signs-cnn.json  - wagi (float32 base64), klasy, przykładowe obrazy do gry
  training/signs_test.json       - wektory testowe (wyjście + gradient) do sprawdzenia JS
"""
import base64
import json
import math
import os
import random
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "frontend", "model")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
S = 48            # rozdzielczość wejścia sieci
SS = 4            # nadpróbkowanie przy rysowaniu
N_PER_CLASS = int(os.environ.get("N_PER_CLASS", "3200"))
EPOCHS = int(os.environ.get("EPOCHS", "9"))

RED, WHITE, BLACK = (200, 16, 46), (246, 246, 242), (20, 20, 20)
YELLOW, BLUE = (245, 196, 0), (0, 82, 160)

CLASSES = [
    ("stop", "STOP"),
    ("yield", "Ustąp pierwszeństwa"),
    ("no_entry", "Zakaz wjazdu"),
    ("no_traffic", "Zakaz ruchu"),
    ("limit30", "Ograniczenie 30"),
    ("limit50", "Ograniczenie 50"),
    ("limit70", "Ograniczenie 70"),
    ("priority", "Droga z pierwszeństwem"),
    ("turn_right", "Nakaz jazdy w prawo"),
    ("crossing", "Przejście dla pieszych"),
    ("danger", "Uwaga, niebezpieczeństwo"),
    ("no_stopping", "Zakaz zatrzymywania"),
]
SHORT = {"stop": "STOP", "yield": "Ustąp", "no_entry": "Zakaz wjazdu", "no_traffic": "Zakaz ruchu",
         "limit30": "30 km/h", "limit50": "50 km/h", "limit70": "70 km/h", "priority": "Pierwszeństwo",
         "turn_right": "W prawo", "crossing": "Przejście", "danger": "Uwaga!", "no_stopping": "Zakaz postoju"}

_fonts = {}


def font(px):
    px = max(6, int(px))
    if px not in _fonts:
        _fonts[px] = ImageFont.truetype(FONT, px)
    return _fonts[px]


def text_center(d, cx, cy, txt, px, fill):
    f = font(px)
    b = d.textbbox((0, 0), txt, font=f)
    d.text((cx - (b[0] + b[2]) / 2, cy - (b[1] + b[3]) / 2), txt, font=f, fill=fill)


def poly(cx, cy, r, n, rot):
    return [(cx + r * math.cos(rot + 2 * math.pi * k / n), cy + r * math.sin(rot + 2 * math.pi * k / n)) for k in range(n)]


def draw_sign(d, cls, cx, cy, r):
    """r = promień znaku w px (dla rozdzielczości SS*S)."""
    if cls == "stop":
        d.polygon(poly(cx, cy, r, 8, math.pi / 8), fill=WHITE)
        d.polygon(poly(cx, cy, r * 0.92, 8, math.pi / 8), fill=RED)
        text_center(d, cx, cy, "STOP", r * 0.62, WHITE)
    elif cls == "yield":
        d.polygon(poly(cx, cy + r * 0.1, r * 1.05, 3, math.pi / 2), fill=RED)
        d.polygon(poly(cx, cy + r * 0.1, r * 0.68, 3, math.pi / 2), fill=YELLOW)
    elif cls == "no_entry":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=RED)
        d.rectangle([cx - r * 0.72, cy - r * 0.17, cx + r * 0.72, cy + r * 0.17], fill=WHITE)
    elif cls in ("no_traffic", "limit30", "limit50", "limit70"):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=RED)
        d.ellipse([cx - r * 0.76, cy - r * 0.76, cx + r * 0.76, cy + r * 0.76], fill=WHITE)
        if cls != "no_traffic":
            text_center(d, cx, cy, cls[-2:], r * 0.95, BLACK)
    elif cls == "priority":
        d.polygon(poly(cx, cy, r * 1.02, 4, 0), fill=BLACK)
        d.polygon(poly(cx, cy, r * 0.97, 4, 0), fill=WHITE)
        d.polygon(poly(cx, cy, r * 0.62, 4, 0), fill=YELLOW)
    elif cls == "turn_right":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=BLUE)
        w = r * 0.17
        d.rectangle([cx - r * 0.6, cy - w, cx + r * 0.15, cy + w], fill=WHITE)
        d.polygon([(cx + r * 0.62, cy), (cx + r * 0.05, cy - r * 0.45), (cx + r * 0.05, cy + r * 0.45)], fill=WHITE)
    elif cls == "crossing":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=BLUE)
        d.polygon([(cx, cy - r * 0.78), (cx - r * 0.8, cy + r * 0.62), (cx + r * 0.8, cy + r * 0.62)], fill=WHITE)
        # pasy i postać
        for k in range(3):
            y = cy + r * (0.42 - 0.0 * k)
            x = cx - r * 0.45 + k * r * 0.32
            d.rectangle([x, y, x + r * 0.18, y + r * 0.1], fill=BLACK)
        d.ellipse([cx - r * 0.09, cy - r * 0.42, cx + r * 0.09, cy - r * 0.24], fill=BLACK)
        d.line([(cx, cy - r * 0.22), (cx, cy + r * 0.1)], fill=BLACK, width=int(r * 0.1))
        d.line([(cx, cy + r * 0.08), (cx - r * 0.16, cy + r * 0.34)], fill=BLACK, width=int(r * 0.09))
        d.line([(cx, cy + r * 0.08), (cx + r * 0.16, cy + r * 0.34)], fill=BLACK, width=int(r * 0.09))
    elif cls == "danger":
        d.polygon(poly(cx, cy - r * 0.08, r * 1.05, 3, -math.pi / 2), fill=RED)
        d.polygon(poly(cx, cy - r * 0.08, r * 0.7, 3, -math.pi / 2), fill=YELLOW)
        text_center(d, cx, cy + r * 0.02, "!", r * 0.7, BLACK)
    elif cls == "no_stopping":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=RED)
        d.ellipse([cx - r * 0.78, cy - r * 0.78, cx + r * 0.78, cy + r * 0.78], fill=BLUE)
        k = r * 0.55
        d.line([(cx - k, cy - k), (cx + k, cy + k)], fill=RED, width=int(r * 0.2))
        d.line([(cx - k, cy + k), (cx + k, cy - k)], fill=RED, width=int(r * 0.2))


def background(rnd, size):
    kind = rnd.random()
    im = Image.new("RGB", (size, size))
    d = ImageDraw.Draw(im)
    if kind < 0.35:   # niebo
        top = np.array([rnd.randint(90, 170), rnd.randint(140, 200), rnd.randint(190, 250)])
        bot = top + rnd.randint(20, 60)
        for y in range(size):
            t = y / size
            d.line([(0, y), (size, y)], fill=tuple(int(v) for v in np.clip(top * (1 - t) + bot * t, 0, 255)))
    elif kind < 0.65:  # zieleń / drzewa
        base = (rnd.randint(30, 90), rnd.randint(70, 140), rnd.randint(30, 80))
        d.rectangle([0, 0, size, size], fill=base)
        for _ in range(rnd.randint(10, 40)):
            x, y, rr = rnd.randint(0, size), rnd.randint(0, size), rnd.randint(size // 20, size // 5)
            c = tuple(int(np.clip(v + rnd.randint(-35, 35), 0, 255)) for v in base)
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=c)
    else:              # miasto: ściany, okna, asfalt
        base = rnd.randint(60, 190)
        d.rectangle([0, 0, size, size], fill=(base, base - rnd.randint(0, 20), base - rnd.randint(0, 25)))
        for _ in range(rnd.randint(3, 12)):
            x0, y0 = rnd.randint(-size // 4, size), rnd.randint(-size // 4, size)
            w, h = rnd.randint(size // 10, size // 2), rnd.randint(size // 10, size // 2)
            g = rnd.randint(30, 220)
            d.rectangle([x0, y0, x0 + w, y0 + h], fill=(g, g + rnd.randint(-20, 20), g + rnd.randint(-20, 20)))
    return im


def render(cls, seed, clean=False):
    rnd = random.Random(seed)
    big = S * SS
    im = background(rnd, big)
    d = ImageDraw.Draw(im)
    if clean:
        r = big * 0.4
        cx = cy = big / 2
    else:
        r = big * rnd.uniform(0.28, 0.46)
        cx = big / 2 + rnd.uniform(-0.12, 0.12) * big
        cy = big / 2 + rnd.uniform(-0.12, 0.12) * big
    # słupek pod znakiem
    if rnd.random() < 0.6:
        d.rectangle([cx - r * 0.07, cy + r * 0.8, cx + r * 0.07, big + 10], fill=(110, 110, 115))
    draw_sign(d, cls, cx, cy, r)
    if not clean:
        if rnd.random() < 0.35:  # przysłonięcia (gałąź, naklejka, brud)
            for _ in range(rnd.randint(1, 3)):
                w, h = rnd.uniform(0.08, 0.3) * r, rnd.uniform(0.08, 0.3) * r
                x, y = cx + rnd.uniform(-0.8, 0.8) * r, cy + rnd.uniform(-0.8, 0.8) * r
                col = tuple(rnd.randint(0, 255) for _ in range(3))
                d.rectangle([x, y, x + w, y + h], fill=col)
        im = im.rotate(rnd.uniform(-11, 11), resample=Image.BILINEAR, center=(cx, cy), fillcolor=im.getpixel((2, 2)))
        if rnd.random() < 0.5:
            sh = rnd.uniform(-0.12, 0.12)
            im = im.transform(im.size, Image.AFFINE, (1, sh, -sh * big / 2, 0, 1, 0), resample=Image.BILINEAR, fillcolor=im.getpixel((2, 2)))
        if rnd.random() < 0.5:
            im = im.filter(ImageFilter.GaussianBlur(rnd.uniform(0.3, 2.5)))
    im = im.resize((S, S), Image.BOX)
    a = np.asarray(im, dtype=np.float32) / 255.0
    if not clean:
        a = a * rnd.uniform(0.6, 1.25) + rnd.uniform(-0.12, 0.12)          # jasność, kontrast
        a = a * np.array([rnd.uniform(0.85, 1.15) for _ in range(3)])     # barwa światła
        a = a + np.random.default_rng(seed).normal(0, rnd.uniform(0, 0.05), a.shape)
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def _job(args):
    return render(*args)


class SignNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1); self.b1 = nn.BatchNorm2d(16)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1); self.b2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1); self.b3 = nn.BatchNorm2d(64)
        self.f1 = nn.Linear(64 * 6 * 6, 128)
        self.drop = nn.Dropout(0.3)
        self.f2 = nn.Linear(128, n)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.b1(self.c1(x))), 2)
        x = F.max_pool2d(F.relu(self.b2(self.c2(x))), 2)
        x = F.max_pool2d(F.relu(self.b3(self.c3(x))), 2)
        x = self.drop(F.relu(self.f1(torch.flatten(x, 1))))
        return self.f2(x)


class Folded(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L

    def forward(self, x):
        (w1, b1), (w2, b2), (w3, b3), (w4, b4), (w5, b5) = self.L
        x = F.max_pool2d(F.relu(F.conv2d(x, w1, b1, padding=1)), 2)
        x = F.max_pool2d(F.relu(F.conv2d(x, w2, b2, padding=1)), 2)
        x = F.max_pool2d(F.relu(F.conv2d(x, w3, b3, padding=1)), 2)
        x = F.relu(F.linear(torch.flatten(x, 1), w4, b4))
        return F.linear(x, w5, b5)


def fold(conv, bn):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return (conv.weight * s[:, None, None, None]).detach(), ((conv.bias - bn.running_mean) * s + bn.bias).detach()


def b64(t):
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t
    return base64.b64encode(np.ascontiguousarray(a.astype("<f4")).tobytes()).decode()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    names = [c for c, _ in CLASSES]
    jobs, ys = [], []
    for ci, c in enumerate(names):
        for k in range(N_PER_CLASS):
            jobs.append((c, ci * 1_000_000 + k)); ys.append(ci)
    vjobs, vys = [], []
    for ci, c in enumerate(names):
        for k in range(300):
            vjobs.append((c, 900_000 + ci * 1_000_000 + k)); vys.append(ci)
    log("renderowanie", len(jobs), "+", len(vjobs))
    with Pool(os.cpu_count()) as p:
        X = np.stack(p.map(_job, jobs, chunksize=128))
        XV = np.stack(p.map(_job, vjobs, chunksize=128))
    Y, YV = torch.tensor(ys), torch.tensor(vys)
    Xt = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()   # uint8 N,3,48,48
    XVt = torch.from_numpy(XV).permute(0, 3, 1, 2).float().div(255)
    log("dane gotowe", Xt.shape)

    model = SignNet(len(names))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    bs = 128
    steps = EPOCHS * math.ceil(len(Xt) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.15)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt))
        t0, tot = time.time(), 0.0
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            xb = Xt[idx].float().div(255)
            xb = (xb + torch.randn_like(xb) * 0.02 * torch.rand(len(idx), 1, 1, 1)).clamp(0, 1)
            loss = F.cross_entropy(model(xb), Y[idx], label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            acc = (model(XVt).argmax(1) == YV).float().mean().item()
        log(f"epoka {ep + 1}/{EPOCHS} loss={tot / len(Xt):.4f} val_acc={acc:.4f} ({time.time() - t0:.0f}s)")

    model.eval()
    L = [fold(model.c1, model.b1), fold(model.c2, model.b2), fold(model.c3, model.b3),
         (model.f1.weight.detach(), model.f1.bias.detach()), (model.f2.weight.detach(), model.f2.bias.detach())]
    fm = Folded(L).eval()
    with torch.no_grad():
        lg = fm(XVt)
    acc = (lg.argmax(1) == YV).float().mean().item()
    best_t, best = 1.0, 1e9
    for t in np.arange(0.5, 3.01, 0.05):
        nll = F.cross_entropy(lg / float(t), YV).item()
        if nll < best:
            best_t, best = float(t), nll
    log(f"folded acc={acc:.4f} T={best_t:.2f}")

    # obrazy do gry: czyste, pewnie rozpoznane
    showcase = []
    for ci, c in enumerate(names):
        got = 0
        for k in range(40):
            img = render(c, 5_000_000 + ci * 1000 + k, clean=True)
            x = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255)
            with torch.no_grad():
                p = F.softmax(fm(x) / best_t, 1)[0]
            if int(p.argmax()) == ci and float(p[ci]) > 0.97:
                showcase.append({"cls": ci, "px": base64.b64encode(img.tobytes()).decode(), "conf": round(float(p[ci]), 4)})
                got += 1
                if got >= 4:
                    break
        log(f"showcase {c}: {got}")

    out = {
        "name": "SignNet", "input": [3, S, S], "temperature": round(best_t, 3), "valAccuracy": round(acc, 4),
        "classes": [{"id": c, "pl": pl, "short": SHORT[c]} for c, pl in CLASSES],
        "layers": [
            {"type": "conv", "in": 3, "out": 16, "w": b64(L[0][0]), "b": b64(L[0][1])},
            {"type": "conv", "in": 16, "out": 32, "w": b64(L[1][0]), "b": b64(L[1][1])},
            {"type": "conv", "in": 32, "out": 64, "w": b64(L[2][0]), "b": b64(L[2][1])},
            {"type": "dense", "in": 2304, "out": 128, "relu": True, "w": b64(L[3][0]), "b": b64(L[3][1])},
            {"type": "dense", "in": 128, "out": len(names), "relu": False, "w": b64(L[4][0]), "b": b64(L[4][1])},
        ],
        "showcase": showcase,
    }
    with open(os.path.join(OUT, "signs-cnn.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

    # wektory testowe: prawdopodobieństwa + gradient log p(cel) po wejściu
    tests = []
    for s in showcase[:6]:
        img = np.frombuffer(base64.b64decode(s["px"]), dtype=np.uint8).reshape(S, S, 3)
        x = torch.from_numpy(img.copy()).permute(2, 0, 1)[None].float().div(255).requires_grad_(True)
        lgx = fm(x)[0]
        p = F.softmax(lgx / best_t, 0)
        tgt = (s["cls"] + 4) % len(names)
        g, = torch.autograd.grad(F.log_softmax(lgx, 0)[tgt], x)
        tests.append({"px": s["px"], "probs": [round(float(v), 6) for v in p], "target": tgt,
                      "grad": base64.b64encode(g[0].numpy().astype("<f4").tobytes()).decode()})
    with open(os.path.join(HERE, "signs_test.json"), "w", encoding="utf-8") as f:
        json.dump(tests, f)
    Image.fromarray(np.concatenate([np.frombuffer(base64.b64decode(s["px"]), dtype=np.uint8).reshape(S, S, 3)
                                    for s in showcase[::4]], 1)).resize((S * 4 * len(showcase[::4]), S * 4), Image.NEAREST
                                                                        ).save(os.path.join(HERE, "signs_preview.png"))
    Image.fromarray(np.concatenate([render(c, 77 + i) for i, c in enumerate(names)], 1)).resize(
        (S * 4 * len(names), S * 4), Image.NEAREST).save(os.path.join(HERE, "signs_train_preview.png"))
    log("GOTOWE")


if __name__ == "__main__":
    main()
