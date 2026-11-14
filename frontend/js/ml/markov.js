// Predyktor kolejnego bitu: mieszanka ekspertów ważona bayesowsko (Bayesian model averaging).
//  - łańcuchy Markowa rzędu 1..4 uczone NA ŻYWO na kliknięciach gracza (n-gramy),
//  - te same n-gramy z ciągów poprzednich odwiedzających (wspólna pamięć stoiska),
//  - ekspert "przekory" (ludzie zmieniają cyfrę częściej niż w 50% przypadków),
//  - ekspert "równowagi" (ludzie wyrównują liczbę zer i jedynek).
// Waga eksperta rośnie wykładniczo z jego trafnością: w_e ∝ Π p_e(prawdziwy bit)^η.

const MAX_K = 4;
const ETA = 1.6;

export class MarkovPredictor {
  constructor(crowd = {}) {
    this.hist = [];
    this.own = new Map();
    this.crowd = crowd;
    this.experts = [];
    for (let k = 1; k <= MAX_K; k++) this.experts.push({ id: `n${k}`, label: `${k}-gram`, p: () => this.ownP(k) });
    for (let k = 1; k <= 3; k++) this.experts.push({ id: `c${k}`, label: `tłum ${k}`, p: () => this.crowdP(k) });
    this.experts.push({ id: 'sw', label: 'przekora', p: () => this.switchP() });
    this.experts.push({ id: 'bal', label: 'równowaga', p: () => this.balanceP() });
    this.logw = this.experts.map((e) => (e.id === 'sw' ? 0.5 : e.id[0] === 'c' ? 0.25 : 0));
    this.lastP = null;
  }

  ctx(k) { return this.hist.length < k ? null : this.hist.slice(-k).join(''); }

  ownP(k) {
    const c = this.ctx(k);
    const t = c === null ? null : this.own.get(c);
    return t ? (t[1] + 0.4) / (t[0] + t[1] + 0.8) : 0.5;
  }

  crowdP(k) {
    const c = this.ctx(k);
    const t = c === null ? null : this.crowd[c];
    return t && t[0] + t[1] >= 6 ? (t[1] + 1) / (t[0] + t[1] + 2) : 0.5;
  }

  switchP() {
    if (!this.hist.length) return 0.5;
    const last = this.hist[this.hist.length - 1];
    let run = 0;
    for (let i = this.hist.length - 1; i >= 0 && this.hist[i] === last; i--) run++;
    const pSwitch = run >= 3 ? 0.85 : run === 2 ? 0.72 : 0.6;
    return last === 0 ? pSwitch : 1 - pSwitch;
  }

  balanceP() {
    const ones = this.hist.reduce((a, b) => a + b, 0);
    return Math.min(0.85, Math.max(0.15, 0.5 - 0.07 * (2 * ones - this.hist.length)));
  }

  /** P(następny bit = 1) wg mieszanki. */
  probOne() {
    const ps = this.experts.map((e) => e.p());
    this.lastP = ps;
    const m = Math.max(...this.logw);
    let num = 0, den = 0;
    ps.forEach((p, i) => { const w = Math.exp(this.logw[i] - m); num += w * p; den += w; });
    return num / den;
  }

  predict() {
    const p = this.probOne();
    return { bit: Math.abs(p - 0.5) < 1e-9 ? (Math.random() < 0.5 ? 0 : 1) : p > 0.5 ? 1 : 0, p };
  }

  update(bit) {
    const ps = this.lastP || this.experts.map((e) => e.p());
    ps.forEach((p, i) => { this.logw[i] += ETA * Math.log(Math.max(1e-3, bit ? p : 1 - p)); });
    for (let k = 1; k <= MAX_K; k++) {
      const c = this.ctx(k);
      if (c === null) continue;
      const t = this.own.get(c) || [0, 0];
      t[bit]++;
      this.own.set(c, t);
    }
    this.hist.push(bit);
    this.lastP = null;
  }

  /** Udział ekspertów w decyzji (do wizualizacji). */
  weights() {
    const m = Math.max(...this.logw);
    const w = this.logw.map((l) => Math.exp(l - m));
    const s = w.reduce((a, b) => a + b, 0);
    return this.experts.map((e, i) => ({ id: e.id, label: e.label, w: w[i] / s }));
  }
}

// ---------------------------------------------------------------------------- entropia Shannona
const H = (ps) => -ps.reduce((a, p) => a + (p > 0 ? p * Math.log2(p) : 0), 0);

/** Entropia blokowa na symbol: H(bloków długości n) / n. Dla idealnego losu = 1 bit. */
export function blockEntropy(seq, n) {
  if (seq.length < n) return null;
  const counts = new Map();
  for (let i = 0; i + n <= seq.length; i++) {
    const k = seq.slice(i, i + n).join('');
    counts.set(k, (counts.get(k) || 0) + 1);
  }
  const tot = seq.length - n + 1;
  return H([...counts.values()].map((c) => c / tot)) / n;
}

/** Entropia warunkowa H(X_t | X_{t-k..t-1}) estymowana z liczności. */
export function conditionalEntropy(seq, k = 2) {
  const t = new Map();
  for (let i = k; i < seq.length; i++) {
    const c = seq.slice(i - k, i).join('');
    const v = t.get(c) || [0, 0];
    v[seq[i]]++;
    t.set(c, v);
  }
  const tot = Math.max(1, seq.length - k);
  let h = 0;
  for (const [c0, c1] of t.values()) {
    const n = c0 + c1;
    h += (n / tot) * H([c0 / n, c1 / n]);
  }
  return t.size ? h : null;
}

/** Liczności przejść łańcucha Markowa rzędu 2: stan "ab" -> następny bit. */
export function transitions(seq) {
  const t = { '00': [0, 0], '01': [0, 0], '10': [0, 0], '11': [0, 0] };
  for (let i = 2; i < seq.length; i++) t[`${seq[i - 2]}${seq[i - 1]}`][seq[i]]++;
  return t;
}

/** Częstości wzorców 3-bitowych vs oczekiwane przy idealnym losie. */
export function trigramStats(seq) {
  const keys = ['000', '001', '010', '011', '100', '101', '110', '111'];
  const c = Object.fromEntries(keys.map((k) => [k, 0]));
  for (let i = 0; i + 3 <= seq.length; i++) c[seq.slice(i, i + 3).join('')]++;
  const exp = Math.max(0, seq.length - 2) / 8;
  return keys.map((k) => ({ k, n: c[k], ratio: exp ? c[k] / exp : 1 }));
}

export function switchRate(seq) {
  let s = 0;
  for (let i = 1; i < seq.length; i++) if (seq[i] !== seq[i - 1]) s++;
  return seq.length > 1 ? s / (seq.length - 1) : 0.5;
}

/** Krótka diagnoza ludzkiego "biasu" (null, jeśli nic wyraźnego). */
export function diagnose(seq) {
  if (seq.length < 8) return null;
  const tri = Object.fromEntries(trigramStats(seq).map((t) => [t.k, t]));
  const sw = switchRate(seq);
  const ones = seq.reduce((a, b) => a + b, 0) / seq.length;
  const runs = (tri['000'].ratio + tri['111'].ratio) / 2;
  if (tri['111'].ratio < 0.35 && tri['000'].ratio < 0.35) return 'Unikasz długich serii!';
  if (tri['111'].ratio < 0.3) return 'Boisz się serii jedynek!';
  if (tri['000'].ratio < 0.3) return 'Boisz się serii zer!';
  if (sw > 0.66) return 'Za często zmieniasz cyfrę!';
  if (sw < 0.36) return 'Za rzadko zmieniasz cyfrę!';
  if (ones > 0.66) return 'Wolisz jedynki!';
  if (ones < 0.34) return 'Wolisz zera!';
  if (runs > 1.8) return 'Lubisz długie serie!';
  const top = Object.values(tri).sort((a, b) => b.ratio - a.ratio)[0];
  if (top.ratio > 2.2) return `Powtarzasz wzorzec ${top.k}!`;
  return null;
}
