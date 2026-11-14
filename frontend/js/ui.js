// Wspólne elementy interfejsu: ikony (kreska jak na ekranie robota), timery, ekrany wyników.

export const reducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

export function el(markup) {
  const t = document.createElement('template');
  t.innerHTML = markup.trim();
  return t.content.firstElementChild;
}

export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

export function plural(n, one, few, many) {
  if (n === 1) return one;
  const d = n % 10, h = n % 100;
  return d >= 2 && d <= 4 && (h < 12 || h > 14) ? few : many;
}

export const pct = (v) => `${Math.round(v * 100)}%`;

/** Zaokrągla rozkład prawdopodobieństwa do pełnych procentów tak, by suma ZAWSZE wynosiła 100 (metoda największych reszt). */
export function pcts(ps) {
  const tot = ps.reduce((a, b) => a + b, 0) || 1;
  const raw = ps.map((p) => (p / tot) * 100);
  const out = raw.map(Math.floor);
  let left = 100 - out.reduce((a, b) => a + b, 0);
  raw.map((r, i) => [r - out[i], i]).sort((a, b) => b[0] - a[0]).forEach(([, i]) => { if (left > 0) { out[i] += 1; left -= 1; } });
  return out;
}

/** Szerokość wypełnienia paska w px: małe wartości jako okrągła kropka mieszcząca się w torze, 0% = brak. */
export function barWidth(fill, v) {
  const track = fill.parentElement;
  return v < 1 ? '0px' : `${Math.max(track.offsetHeight, (v / 100) * track.offsetWidth)}px`;
}
export const secs = (ms) => (ms / 1000).toLocaleString('pl-PL', { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + ' s';

export function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/** Sleep przerywany przez AbortSignal (rozwiązuje się false przy przerwaniu). */
export function wait(ms, signal) {
  return new Promise((res) => {
    if (signal?.aborted) return res(false);
    const t = setTimeout(() => res(true), ms);
    signal?.addEventListener('abort', () => { clearTimeout(t); res(false); }, { once: true });
  });
}

/** Timer oparty o requestAnimationFrame. onTick(progress 0..1, msLeft). */
export function countdown(ms, { onTick, onEnd, signal } = {}) {
  const start = performance.now();
  let raf = 0, done = false;
  const frame = (now) => {
    if (done) return;
    const t = Math.min(1, (now - start) / ms);
    onTick?.(t, Math.max(0, ms - (now - start)));
    if (t >= 1) { done = true; onEnd?.(); return; }
    raf = requestAnimationFrame(frame);
  };
  raf = requestAnimationFrame(frame);
  const stop = () => { done = true; cancelAnimationFrame(raf); };
  signal?.addEventListener('abort', stop, { once: true });
  return { stop, elapsed: () => performance.now() - start };
}

/** Pierścień odliczania (SVG). */
export function ring(size = 44) {
  const r = 25, c = 2 * Math.PI * r;
  const node = el(`
    <div class="ring" style="--size:${size}px">
      <svg viewBox="0 0 60 60" aria-hidden="true">
        <circle class="ring__track" cx="30" cy="30" r="${r}"/>
        <circle class="ring__bar" cx="30" cy="30" r="${r}" stroke-dasharray="${c}" stroke-dashoffset="0"/>
      </svg>
      <span class="ring__num"></span>
    </div>`);
  const bar = node.querySelector('.ring__bar');
  const num = node.querySelector('.ring__num');
  return {
    node,
    set(progress, msLeft) {
      bar.style.strokeDashoffset = String(c * progress);
      num.textContent = String(Math.ceil(msLeft / 1000));
      node.classList.toggle('is-low', msLeft < 5000);
    },
  };
}

const svg = (body) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;

export const icons = {
  back: svg('<path d="M14.5 5.5L8 12l6.5 6.5"/>'),
  replay: svg('<path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3M4.5 4.5v4h4"/>'),
  home: svg('<path d="M4 11l8-6.5 8 6.5M6.5 9.5V19h11V9.5"/>'),
  play: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8.5 6v12l9.5-6z" fill="currentColor"/></svg>',
  check: svg('<path d="M5 12.5l4.5 4.5L19 7.5"/>'),
  cross: svg('<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>'),
  mark: `<svg viewBox="0 0 40 40" aria-hidden="true"><g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M10 12 L20 20 L10 28 M10 12 L10 28 M20 20 H31"/></g><circle cx="10" cy="12" r="3.2" fill="currentColor"/><circle cx="10" cy="28" r="3.2" fill="currentColor"/><circle cx="20" cy="20" r="3.2" fill="currentColor"/><circle cx="31" cy="20" r="3.6" fill="var(--route)"/></svg>`,
};

/**
 * Ekran wyniku. Zwraca Promise<'again'|'home'>.
 * opts: { score, scoreLabel, headline, body (Node|string), tone: 'go'|'stop'|'idle' }
 */
export function resultScreen(stage, opts) {
  return new Promise((resolve) => {
    const node = el(`
      <div class="result enter" data-tone="${opts.tone || 'idle'}">
        <div class="result__score">
          <span class="result__big">${opts.score}</span>
          ${opts.scoreLabel ? `<span class="result__label">${esc(opts.scoreLabel)}</span>` : ''}
        </div>
        <h2 class="result__headline">${esc(opts.headline)}</h2>
        <div class="result__body"></div>
        <div class="result__actions">
          <button class="btn btn--primary" data-act="again">${icons.replay}<span>Zagraj ponownie</span></button>
          <button class="btn btn--ghost" data-act="home">${icons.home}<span>Menu</span></button>
        </div>
      </div>`);
    const body = node.querySelector('.result__body');
    if (opts.body instanceof Node) body.append(opts.body);
    else if (opts.body) body.innerHTML = opts.body;
    else body.remove();
    node.addEventListener('click', (e) => {
      const b = e.target.closest('[data-act]');
      if (b) resolve(b.dataset.act);
    });
    stage.replaceChildren(node);
  });
}

/** Krótki komunikat na środku elementu (np. "Ty pierwszy!"). */
export function flash(container, markup, tone = 'go', ms = 1300) {
  const n = el(`<div class="flash" data-tone="${tone}">${markup}</div>`);
  container.append(n);
  setTimeout(() => n.classList.add('is-out'), ms);
  setTimeout(() => n.remove(), ms + 400);
  return n;
}
