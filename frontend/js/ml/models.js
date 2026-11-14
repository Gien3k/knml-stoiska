import { MLP } from './mlp.js';

let digitsPromise = null;

function b64ToF32(b64) {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}

/** Sieć MLP do cyfr z kropek + poziomy gry "Złudzenie sieci" (training/train_digits.py). */
export function loadDigits() {
  digitsPromise ||= fetch('model/digits-mlp.json')
    .then((r) => { if (!r.ok) throw new Error('digits'); return r.json(); })
    .then((j) => ({
      net: new MLP(j),
      G: j.grid,
      names: j.names,
      acc: j.acc,
      maxChanges: j.maxChanges,
      levels: j.levels.map((l) => ({ ...l, x: Float32Array.from(l.x, (c) => (c === '1' ? 1 : 0)) })),
    }))
    .catch((e) => { digitsPromise = null; throw e; });
  return digitsPromise;
}

export function preloadModels() {
  loadDigits().catch(() => {});
}
