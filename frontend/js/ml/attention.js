// Model uwagi systemu wizyjnego dla gry "Oszukaj Autopilot".
// Deterministyczny i ciągły: każdy znak ma 3-4 strefy wysokiej aktywacji (2 kluczowe + 2 poboczne).
// Naklejka w punkcie p tłumi strefę k o g = exp(-|p - h_k|² / 2τ²) (gaussowsko), a obie naklejki razem
// tłumią strefę jak 1 - Π(1 - g). Logity:
//   z_źródło = A - Σ α_k S_k
//   z_cel    = B + Σ β_k S_k + γ · S_klucz1 · S_klucz2      (nieliniowa interakcja dwóch naklejek)
// Klasa wyjściowa = softmax(z). Mapa uwagi (CAM) = Σ α_k (1 - S_k) · N(h_k, σ_k).

const A = 4.6, B = 0.7, GAMMA = 2.4, TAU = 0.085;
const OTHER = [0, -0.5, -1];            // pozostałe klasy detektora (nie pokazywane)
const W = { key: { a: 2.2, b: 1.6 }, minor: { a: 0.8, b: 0.5 } };

export const SCENARIOS = [
  {
    id: 'stop', sign: 'stop', tgtSign: 'limit100',
    goal: ['Przyklej 2 naklejki na znaku', 'STOP', ', aby Autopilot przyspieszył do', '100 km/h', '!'],
    src: { name: 'ZNAK STOP', act: 'HAMOWANIE', tag: 'STOP' },
    tgt: { name: 'OGRANICZENIE 100 km/h', act: 'PRZYSPIESZAM', tag: '100 km/h' },
    win: 'Auto widzi 100 km/h',
    hot: [{ x: 0.26, y: 0.5, s: 0.1, key: true }, { x: 0.74, y: 0.5, s: 0.1, key: true }, { x: 0.5, y: 0.12, s: 0.11 }, { x: 0.18, y: 0.8, s: 0.1 }],
  },
  {
    id: 'limit50', sign: 'limit50', tgtSign: 'noentry',
    goal: ['Przyklej 2 naklejki na znaku', '50 km/h', ', aby Autopilot zobaczył', 'zakaz wjazdu', '!'],
    src: { name: 'OGRANICZENIE 50 km/h', act: 'JADĘ 50 km/h', tag: '50 km/h' },
    tgt: { name: 'ZAKAZ WJAZDU', act: 'HAMOWANIE AWARYJNE', tag: 'ZAKAZ WJAZDU' },
    win: 'Auto widzi zakaz wjazdu',
    hot: [{ x: 0.37, y: 0.5, s: 0.1, key: true }, { x: 0.64, y: 0.5, s: 0.1, key: true }, { x: 0.5, y: 0.09, s: 0.1 }, { x: 0.91, y: 0.56, s: 0.1 }],
  },
  {
    // klasyczny atak na auta autonomiczne: zakrycie lewych "otworów" cyfry 3 sprawia, że wygląda jak 8
    id: 'limit30', sign: 'limit30', tgtSign: 'limit80',
    goal: ['Przyklej 2 naklejki na znaku', '30 km/h', ', aby autopilot zobaczył', '80 km/h', '!'],
    src: { name: 'OGRANICZENIE 30 km/h', act: 'JADĘ 30 km/h', tag: '30 km/h' },
    tgt: { name: 'OGRANICZENIE 80 km/h', act: 'PRZYSPIESZAM DO 80', tag: '80 km/h' },
    win: 'Auto widzi 80 km/h',
    hot: [{ x: 0.34, y: 0.4, s: 0.1, key: true }, { x: 0.34, y: 0.61, s: 0.1, key: true }, { x: 0.6, y: 0.5, s: 0.1 }, { x: 0.5, y: 0.09, s: 0.09 }],
  },
  {
    id: 'crossing', sign: 'crossing', tgtSign: 'parking',
    goal: ['Przyklej 2 naklejki na znaku', 'przejścia dla pieszych', ', aby autopilot zobaczył', 'parking', '!'],
    src: { name: 'PRZEJŚCIE DLA PIESZYCH', act: 'ZWALNIAM', tag: 'PRZEJŚCIE' },
    tgt: { name: 'PARKING', act: 'JADĘ BEZ ZWALNIANIA', tag: 'PARKING' },
    win: 'Auto widzi parking',
    hot: [{ x: 0.5, y: 0.47, s: 0.1, key: true }, { x: 0.5, y: 0.75, s: 0.1, key: true }, { x: 0.5, y: 0.2, s: 0.09 }, { x: 0.12, y: 0.12, s: 0.09 }],
  },
  {
    id: 'right', sign: 'right', tgtSign: 'left',
    goal: ['Przyklej 2 naklejki na znaku', 'nakaz jazdy w prawo', ', aby Autopilot skręcił', 'w lewo', '!'],
    src: { name: 'NAKAZ JAZDY W PRAWO', act: 'SKRĘCAM W PRAWO', tag: 'W PRAWO' },
    tgt: { name: 'NAKAZ JAZDY W LEWO', act: 'SKRĘCAM W LEWO', tag: 'W LEWO' },
    win: 'Auto skręca w lewo',
    hot: [{ x: 0.7, y: 0.5, s: 0.1, key: true }, { x: 0.32, y: 0.5, s: 0.1, key: true }, { x: 0.5, y: 0.1, s: 0.09 }, { x: 0.5, y: 0.9, s: 0.09 }],
  },
];

/** stickers: [{x, y} | null] we współrzędnych znaku (0..1). */
export function evaluate(sc, stickers) {
  const S = sc.hot.map((h) => {
    let keep = 1;
    for (const s of stickers) {
      if (!s) continue;
      const d2 = (s.x - h.x) ** 2 + (s.y - h.y) ** 2;
      keep *= 1 - Math.exp(-d2 / (2 * TAU * TAU));
    }
    return 1 - keep;
  });
  let zs = A, zt = B;
  const keys = [];
  sc.hot.forEach((h, k) => {
    const w = h.key ? W.key : W.minor;
    zs -= w.a * S[k];
    zt += w.b * S[k];
    if (h.key) keys.push(S[k]);
  });
  zt += GAMMA * keys[0] * keys[1];
  const z = [zs, zt, ...OTHER];
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const sum = e.reduce((a, b) => a + b, 0);
  // siła zakłócenia każdej naklejki: o ile procent spada łączna uwaga sieci przez tę naklejkę
  const tot = sc.hot.reduce((a, h) => a + (h.key ? W.key.a : W.minor.a), 0);
  const drop = stickers.map((s) => {
    if (!s) return 0;
    return sc.hot.reduce((a, h) => {
      const d2 = (s.x - h.x) ** 2 + (s.y - h.y) ** 2;
      return a + (h.key ? W.key.a : W.minor.a) * Math.exp(-d2 / (2 * TAU * TAU));
    }, 0) / tot;
  });
  return { pSrc: e[0] / sum, pTgt: e[1] / sum, S, drop };
}

/**
 * Rysuje pulsującą mapę uwagi (w stylu Grad-CAM). size = bok kwadratu znaku w pikselach płótna.
 * pad = margines wokół znaku (ułamek size), żeby poświata wygasała płynnie i nie była ucinana krawędzią płótna.
 */
export function drawHeat(g, sc, stickers, size, t, S, { hint = 0, pad = 0 } = {}) {
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, g.canvas.width, g.canvas.height);
  g.translate(pad * size, pad * size);
  g.globalCompositeOperation = 'lighter';
  const pulse = 0.82 + 0.18 * Math.sin(t / 380);
  sc.hot.forEach((h, k) => {
    const w = h.key ? 1 : 0.55;
    const inten = w * (1 - S[k]) * pulse;
    if (inten < 0.02) return;
    // strefa "odpycha się" od naklejek
    let px = h.x, py = h.y;
    for (const s of stickers) {
      if (!s) continue;
      const dx = h.x - s.x, dy = h.y - s.y, d = Math.hypot(dx, dy) || 1e-3;
      const f = Math.exp(-(d * d) / (2 * 0.12 * 0.12)) * 0.06;
      px += (dx / d) * f; py += (dy / d) * f;
    }
    const r = size * Math.min(0.24, h.s * (2.3 - S[k]));
    const gr = g.createRadialGradient(px * size, py * size, 0, px * size, py * size, r);
    gr.addColorStop(0, `rgba(255,245,150,${(0.95 * inten).toFixed(3)})`);
    gr.addColorStop(0.3, `rgba(255,170,40,${(0.8 * inten).toFixed(3)})`);
    gr.addColorStop(0.65, `rgba(255,70,30,${(0.4 * inten).toFixed(3)})`);
    gr.addColorStop(1, 'rgba(255,40,20,0)');
    g.fillStyle = gr;
    g.beginPath(); g.arc(px * size, py * size, r, 0, Math.PI * 2); g.fill();
    if (hint && h.key) {
      g.globalCompositeOperation = 'source-over';
      g.strokeStyle = `rgba(243,241,234,${(0.35 + 0.35 * Math.sin(t / 200)) * hint})`;
      g.lineWidth = size * 0.008;
      g.setLineDash([size * 0.02, size * 0.015]);
      g.beginPath(); g.arc(h.x * size, h.y * size, size * 0.09, 0, Math.PI * 2); g.stroke();
      g.setLineDash([]);
      g.globalCompositeOperation = 'lighter';
    }
  });
  g.globalCompositeOperation = 'source-over';
  g.setTransform(1, 0, 0, 1, 0, 0);
}
