// Wielowarstwowy perceptron (MLP): 196 -> 96 -> 48 -> 10, ReLU, softmax.
// Wagi wytrenowane w PyTorch na cyfrach MNIST (training/train_digits.py).

function b64ToF32(b64) {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}

export class MLP {
  constructor(json) {
    this.layers = json.layers.map((l) => ({ ...l, w: b64ToF32(l.w), b: b64ToF32(l.b) }));
  }

  /** x: Float32Array(196) z wartościami 0/1. Zwraca prawdopodobieństwa 10 cyfr (softmax z temperaturą T). */
  predict(x, T = 1) {
    let a = x;
    this.layers.forEach((L, k) => {
      const out = new Float32Array(L.out);
      for (let o = 0; o < L.out; o++) {
        let s = L.b[o];
        const wb = o * L.in;
        for (let i = 0; i < L.in; i++) s += L.w[wb + i] * a[i];
        out[o] = k < this.layers.length - 1 && s < 0 ? 0 : s;
      }
      a = out;
    });
    const m = Math.max(...a);
    const e = Array.from(a, (v) => Math.exp((v - m) / T));
    const s = e.reduce((p, q) => p + q, 0);
    return e.map((v) => v / s);
  }
}
