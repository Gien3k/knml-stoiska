"""
Poziomy gry "Zhakuj pismo ręczne" - wersja "ludzie vs AI" (można przegrać).

Gracz widzi 8-10 pulsujących punktów na cyfrze z MNIST i ma 5 ruchów BEZ cofania. Wygrywa poziom,
gdy NAJWYŻSZA odpowiedź sieci (argmax, czyli decyzja klasyfikatora) to cyfra-cel.
Tylko 4 punkty naprawdę działają, reszta to wabiki (poziom 1: 4 z 8, kolejne: 4 z 10). Punkt to dołożenie
kropki obok kreski albo zabranie kropki z kreski - oba rodzaje bywają dobre i bywają wabikami.
Dla par 7->9, 9->8, 5->6, 3->8, 0->8 dobieramy cyfrę i punkty tak, żeby (sprawdzane wyczerpująco
na prawdziwej sieci MLP, na logitach - niezależnie od temperatury):
  - KAŻDE 3 dobre punkty + do 2 wabików (stan w chwili trzeciego trafienia): cel wygrywa z zapasem MARGIN,
  - KAŻDY stan z co najwyżej 2 dobrymi punktami (i dowolnymi wabikami, razem <= 5 ruchów):
    cel przegrywa z zapasem MARGIN - mniej niż 3 trafienia nigdy nie wystarczą.
Temperaturę softmaxu T (tylko do wyświetlania pewności) dobieramy tak, żeby każdy dobry punkt sam
podnosił p(cel) o >= JUMP_GOOD (widać po kliknięciu, że pomógł), a wabik zmieniał ją najwyżej
o DECOY_MAX (widać, że nie pomógł), i żeby zaokrąglone procenty nie dawały remisu z liderem.
Losowe klikanie wygrywa poziom z prawdopodobieństwem P(>= 3 dobre w 5 losowych):
  4 z 8 -> 50%, 4 z 10 -> 26,2%; całą grę (3 z 5 poziomów) losowo wygrywa się w ok. 14%.
Wynik: nadpisuje "levels" i "maxChanges" w frontend/model/digits-mlp.json (wagi sieci bez zmian).
"""
import base64
import gzip
import itertools
import json
import os
import random
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "frontend", "model", "digits-mlp.json")
MNIST = os.path.join(HERE, "data", "mnist")
G = 14
MOVES = 5
NEED = 3
TIERS = [(7, 9, 4, 8), (9, 8, 4, 10), (5, 6, 4, 10), (3, 8, 4, 10), (0, 8, 4, 10)]   # (cyfra, cel, dobre punkty, wszystkie punkty)
TEMPS = sorted(set(np.round(np.geomspace(5.0, 0.25, 48), 3).tolist()), reverse=True)
MARGIN = 0.25                     # zapas logitu: cel wyraźnie wygrywa / wyraźnie przegrywa z liderem
JUMP_GOOD, DECOY_MAX = 0.035, 0.03
GAP_PTS = 0.02                    # różnica cel-lider w wyświetlanych procentach (bez remisów po zaokrągleniu)
PER_TIER = 6


def idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    nd = data[3]
    dims = [int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(nd)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * nd).reshape(dims)


def main():
    j = json.load(open(MODEL, encoding="utf-8"))
    L = [(np.frombuffer(base64.b64decode(l["w"]), dtype="<f4").reshape(l["out"], l["in"]),
          np.frombuffer(base64.b64decode(l["b"]), dtype="<f4")) for l in j["layers"]]

    def logits(X):
        a = X
        for k, (w, b) in enumerate(L):
            a = a @ w.T + b
            if k < len(L) - 1:
                a = np.maximum(a, 0)
        return a

    def soft(z, T):
        z = z / T
        z = z - z.max(-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(-1, keepdims=True)

    def margin(z, t):
        o = np.delete(z, t, axis=-1)
        return z[..., t] - o.max(-1)

    def apply(x, sets):
        B = np.repeat(x[None], len(sets), 0)
        for r, s in enumerate(sets):
            s = list(s)
            if s:
                B[r, s] = 1 - B[r, s]
        return B

    imgs = idx(os.path.join(MNIST, "t10k-images-idx3-ubyte.gz")).astype(np.float32) / 255
    labels = idx(os.path.join(MNIST, "t10k-labels-idx1-ubyte.gz"))
    X = (imgs.reshape(-1, G, 2, G, 2).mean(axis=(2, 4)) > 0.45).astype(np.float32).reshape(-1, G * G)
    Z = logits(X)
    rng = random.Random(2026)
    out = []
    p_levels = []
    for tier, (src, tgt, ng, nclick) in enumerate(TIERS, start=1):
        nd = nclick - ng
        found = []
        ids = [n for n in range(len(X)) if labels[n] == src]
        rng.shuffle(ids)
        tried = 0
        cnt = dict(sieve=0, decoys=0, screen=0)
        for n in ids:
            x = X[n]
            if Z[n].argmax() != src or soft(Z[n], 1)[src] < 0.9 or not (16 <= x.sum() <= 40):
                continue
            tried += 1
            if tried > 1000 or len(found) >= 14:
                break
            grid = x.reshape(G, G)
            adj = [r * G + c for r in range(G) for c in range(G)
                   if grid[r, c] == 0 and grid[max(0, r - 1):r + 2, max(0, c - 1):c + 2].sum() > 0]
            ink = [i for i in range(G * G) if x[i] == 1]
            za = logits(apply(x, [[i] for i in adj]))
            gain = margin(za, tgt) - margin(Z[n], tgt)      # o ile piksel przybliża cel do lidera
            zr = logits(apply(x, [[i] for i in ink]))
            mr = margin(zr, tgt) - margin(Z[n], tgt)
            # dobre punkty: najsilniejsze zmiany - dołożenie ALBO zabranie kropki (zabranie nie jest zawsze wabikiem)
            both = sorted([(float(g), i) for g, i in zip(gain, adj)] + [(float(m), i) for m, i in zip(mr, ink)], reverse=True)
            order = [i for _, i in both]
            pool_good = order[:8]
            # kandydaci na wabiki: kropki przy kresce (do dołożenia) i na kresce (do zabrania) o małym wpływie na cel
            cand = ([(abs(float(g)), i) for g, i in zip(gain, adj) if i not in order[:12]]
                    + [(abs(float(mr[k])), i) for k, i in enumerate(ink) if i not in order[:12]])
            pool_dec = [i for _, i in sorted(cand)[:24]]
            if len(pool_dec) < nd:
                continue
            # szybkie sito: tylko dobre punkty (bez wabików) - 3 trafienia muszą wygrywać, 2 przegrywać
            subs = [frozenset(s) for k in range(NEED + 1) for s in itertools.combinations(pool_good, k)]
            row = {s: r for r, s in enumerate(subs)}
            zs = logits(apply(x, subs))
            mz = margin(zs, tgt)
            cands = []
            for Gset in itertools.combinations(pool_good, ng):
                tr = [row[frozenset(s)] for s in itertools.combinations(Gset, NEED)]
                lo = [row[frozenset(s)] for k in range(NEED) for s in itertools.combinations(Gset, k)]
                if mz[tr].min() >= MARGIN + 0.15 and mz[lo].max() <= -MARGIN - 0.15:
                    cands.append((Gset, float(mz[tr].min() - mz[lo].max())))
            cands.sort(key=lambda c: -c[1])
            cnt["sieve"] += bool(cands)
            dec_ok = scr_ok = False
            best = None
            for Gset, _ in cands[:12]:
                # ryzyko wabika w kontekście: czy z 2 dobrymi pomaga celowi (mógłby dać wygraną bez 3 trafień),
                # albo czy z 3 dobrymi mu szkodzi (mógłby odebrać zasłużoną wygraną)
                ctx = [frozenset(c) for k in range(NEED + 1) for c in itertools.combinations(Gset, k)]
                base = margin(logits(apply(x, ctx)), tgt)
                zc = margin(logits(apply(x, [set(c) | {d} for d in pool_dec for c in ctx])), tgt).reshape(len(pool_dec), len(ctx))
                delta = zc - base[None]
                small = [r for r, c in enumerate(ctx) if len(c) < NEED]
                full = [r for r, c in enumerate(ctx) if len(c) == NEED]
                risk = np.maximum(delta[:, small].max(1), (-delta[:, full]).max(1))
                ranked = [pool_dec[r] for r in np.argsort(risk)]
                decoy_sets = [ranked[:nd]] + [ranked[:nd - 1] + [ranked[nd + k]] for k in range(min(4, len(ranked) - nd))]
                for D in decoy_sets:
                    win_sets = [set(g3) | set(dd) for g3 in itertools.combinations(Gset, NEED)
                                for k in range(MOVES - NEED + 1) for dd in itertools.combinations(D, k)]
                    lose_sets = [set(gg) | set(dd) for kg in range(NEED) for gg in itertools.combinations(Gset, kg)
                                 for kd in range(MOVES - kg + 1) for dd in itertools.combinations(D, kd)]
                    zw, zl = logits(apply(x, win_sets)), logits(apply(x, lose_sets))
                    mw, ml = margin(zw, tgt).min(), margin(zl, tgt).max()
                    if mw < MARGIN or ml > -MARGIN:
                        continue
                    dec_ok = True
                    zg, zd = logits(apply(x, [[i] for i in Gset])), logits(apply(x, [[i] for i in D]))
                    # najostrzejsza temperatura, przy której każdy dobry punkt wciąż widocznie podnosi pasek
                    for T in sorted(TEMPS):
                        p0 = soft(Z[n][None], T)[0]
                        if p0.argmax() != src or p0[tgt] > 0.12:
                            continue
                        jg = soft(zg, T)[:, tgt] - p0[tgt]
                        jd = np.abs(soft(zd, T)[:, tgt] - p0[tgt])
                        if jg.min() < JUMP_GOOD or jd.max() > DECOY_MAX:
                            continue
                        pw, pl = soft(zw, T), soft(zl, T)
                        gw = (pw[:, tgt] - np.delete(pw, tgt, 1).max(1)).min()
                        gl = (np.delete(pl, tgt, 1).max(1) - pl[:, tgt]).min()
                        if gw < GAP_PTS or gl < GAP_PTS:
                            continue
                        scr_ok = True
                        score = float(jg.min()) + 0.1 * float(mw - ml)
                        if best is None or score > best[0]:
                            best = (score, list(Gset), D, T, float(pw[:, tgt].min()), float(pl[:, tgt].max()),
                                    jg.tolist(), float(p0[src]))
                        break
                if best and best[0] > 0.25:
                    break
            cnt["decoys"] += dec_ok
            cnt["screen"] += scr_ok
            if not best:
                continue
            _, good, D, T, pw, pl, jg, p0src = best
            found.append({"src": src, "tgt": tgt, "T": T, "good": good, "decoys": D,
                          "click": sorted(good + D), "x": "".join(str(int(v)) for v in x),
                          "p0src": round(p0src, 3), "jump": [round(v, 3) for v in jg],
                          "winMin": round(pw, 3), "loseMax": round(pl, 3), "tier": tier, "sol": good[:NEED]})
        # preferuj poziomy, w których zabranie kropki też bywa dobrym ruchem
        found.sort(key=lambda l: -(min(l["jump"]) + 0.5 * (l["winMin"] - l["loseMax"])
                                   + (0.05 if any(l["x"][i] == "1" for i in l["good"]) else 0)))
        keep = found[:PER_TIER]
        p_rand = sum(comb(ng, k) * comb(nd, MOVES - k) for k in range(NEED, min(ng, MOVES) + 1)) / comb(nclick, MOVES)
        p_levels.append(p_rand)
        print(f"poziom {tier} ({src}->{tgt}, {ng} dobre z {nclick}): {len(found)} kandydatów z {tried} cyfr, "
              f"losowe klikanie wygrywa {p_rand:.1%}; etapy: {cnt}")
        for l in keep[:3]:
            print(f"   T={l['T']} start {l['p0src']:.0%}, p(cel) przy 3 trafieniach >= {l['winMin']:.0%}, "
                  f"przy <=2 <= {l['loseMax']:.0%}, skoki {[f'{v:+.0%}' for v in l['jump']]}")
        if not keep:
            raise SystemExit(f"brak poziomów dla {src}->{tgt}")
        out += keep
    # szansa wygrania całej gry (>= 3 z 5 poziomów) losowym klikaniem
    dist = [1.0]
    for p in p_levels:
        dist = [a * (1 - p) + (dist[k - 1] * p if k else 0) for k, a in enumerate(dist + [0.0])]
    print(f"losowe klikanie wygrywa całą grę (>= 3 z 5): {sum(dist[3:]):.1%}")
    j["levels"] = out
    j["maxChanges"] = MOVES
    with open(MODEL, "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False, separators=(",", ":"))
    print("zapisano", len(out), "poziomów")


if __name__ == "__main__":
    main()
