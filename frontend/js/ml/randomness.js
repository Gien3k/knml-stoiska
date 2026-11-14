// Statystyki ciągów 0/1 i generatory: prawdziwie losowy oraz "ludzki" (model z badań psychologicznych:
// ludzie zmieniają cyfrę zbyt często, unikają długich serii i pilnują równowagi zer i jedynek).

export function longestRun(seq) {
  let best = { len: 0, start: 0 }, len = 0, start = 0;
  for (let i = 0; i < seq.length; i++) {
    if (i > 0 && seq[i] === seq[i - 1]) len++;
    else { len = 1; start = i; }
    if (len > best.len) best = { len, start };
  }
  return best;
}

export function switches(seq) {
  let s = 0;
  for (let i = 1; i < seq.length; i++) if (seq[i] !== seq[i - 1]) s++;
  return s;
}

/** Prawdziwie losowy ciąg (kryptograficzny generator liczb losowych). */
export function randomSeq(n = 20) {
  const a = new Uint8Array(n);
  crypto.getRandomValues(a);
  return Array.from(a, (v) => v & 1);
}

/** Ciąg w stylu człowieka. */
export function humanLikeSeq(n = 20) {
  const s = [Math.random() < 0.5 ? 0 : 1];
  let run = 1;
  for (let i = 1; i < n; i++) {
    const last = s[i - 1];
    const pSwitch = run === 1 ? 0.58 : run === 2 ? 0.8 : 0.97;
    let next = Math.random() < pSwitch ? 1 - last : last;
    const ones = s.reduce((a, b) => a + b, 0);
    const diff = ones - (s.length - ones);
    if (Math.abs(diff) >= 3 && Math.random() < 0.6) next = diff > 0 ? 0 : 1;
    run = next === last ? run + 1 : 1;
    s.push(next);
  }
  return s;
}
