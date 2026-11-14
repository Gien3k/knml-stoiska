// Gra 2: "Zhakuj pismo ręczne" - człowiek kontra sieć MLP rozpoznająca cyfry z MNIST.
// Na cyfrze pulsuje 8-10 punktów, ale tylko 4 naprawdę działają (reszta to wabiki). Gracz ma 5 ruchów
// BEZ cofania. Poziom wygrany, gdy najwyższa odpowiedź sieci (jej decyzja) to cyfra-cel.
// Gwarancje sprawdzone offline na prawdziwej sieci (training/make_levels.py): dowolne 3 dobre punkty
// wygrywają (nawet z 2 wabikami), mniej niż 3 nigdy. Po każdym kliknięciu widać, czy pomogło (+x%).
// Gra ma 5 poziomów; człowiek wygrywa grę przy 3 z 5, w przeciwnym razie wygrywa AI.
import { el, resultScreen, flash, shuffle, pcts } from '../ui.js';
import { loadDigits } from '../ml/models.js';
import { cardMarkup, cardRender } from './ilcard.js';

const LEVELS = 5;
const NEED_WINS = 3;

export function mount(stage, ctx) {
  (async () => {
    stage.replaceChildren(el(`<div class="note-center loading">Ładowanie sieci neuronowej…</div>`));
    let D;
    try { D = await loadDigits(); } catch {
      stage.replaceChildren(el(`<div class="note-center">Nie udało się wczytać modelu. Odśwież stronę.</div>`));
      return;
    }
    while (!ctx.signal.aborted) {
      const res = await play(stage, ctx, D);
      if (!res || ctx.signal.aborted) return;
      ctx.setHud(''); ctx.setGoal(null);
      const act = await resultScreen(stage, res);
      if (act === 'home') return ctx.home();
    }
  })();
}

function pickLevels(D) {
  return Array.from({ length: LEVELS }, (_, k) => shuffle(D.levels.filter((l) => l.tier === k + 1))[0]);
}

const hudMarkup = (cur, results) => `<div class="dots">${Array.from({ length: LEVELS }, (_, i) => {
  const r = results[i];
  return `<span class="dot ${r === true ? 'ok' : r === false ? 'bad' : i === cur ? 'cur' : ''}"></span>`;
}).join('')}</div><div class="pts"><b>${Math.min(cur + 1, LEVELS)}</b><span>/ ${LEVELS}</span></div>`;

const argmax = (p) => p.indexOf(Math.max(...p));

async function play(stage, ctx, D) {
  const { net, G } = D;
  const N = G * G;
  const MOVES = D.maxChanges || 5;
  const levels = pickLevels(D);
  const results = [];

  const node = el(`
    <div class="il il--v5 il--duel enter">
      <div class="il__main">
        <div class="il__grid panel"><div class="il__dots" style="--g:${G}">${Array.from({ length: N }, (_, i) => `<button class="dt" data-i="${i}" tabindex="-1"><i></i></button>`).join('')}</div></div>
        ${cardMarkup({ moves: MOVES })}
      </div>
    </div>`);
  stage.replaceChildren(node);
  const $ = (s) => node.querySelector(s);
  const dots = [...node.querySelectorAll('.dt')];
  const grid = $('.il__grid');
  const action = $('.il__action');
  const card = $('.il__card');
  const delta = $('.il__delta');
  let deltaTimer = 0;

  for (let lv = 0; lv < LEVELS; lv++) {
    const L = levels[lv];
    const x = Float32Array.from(L.x);
    const clickable = new Set(L.click);
    const good = new Set(L.good);
    const changed = new Set();
    let p = null, pr = null, state = 'play';      // play | won | lost
    ctx.setHud(hudMarkup(lv, results));
    ctx.setGoal(`Dołóż lub zabierz max ${MOVES} kropek, aby AI zobaczyło: <b>${L.tgt}</b>`);
    node.classList.remove('is-won', 'is-lost');
    grid.classList.remove('swap-in'); void grid.offsetWidth; grid.classList.add('swap-in');
    action.className = 'il__action is-wait';
    action.disabled = true;
    action.textContent = 'Ruchów nie można cofnąć';
    delta.textContent = '';

    const compute = () => { p = net.predict(x, L.T); };
    const canClick = (i) => state === 'play' && clickable.has(i) && !changed.has(i) && changed.size < MOVES;
    const render = () => {
      pr = pcts(Array.from(p));   // pełne procenty, zawsze razem 100%
      const top = argmax(p);
      dots.forEach((d, i) => {
        d.className = 'dt';
        if (x[i] === 1) d.classList.add('on');
        if (clickable.has(i) && !changed.has(i) && state === 'play') d.classList.add('can');
        if (changed.has(i)) d.classList.add(L.x[i] === 0 ? 'added' : 'removed');
        if (state === 'lost' && good.has(i) && !changed.has(i)) d.classList.add('sol');
      });
      card.dataset.state = top === L.tgt ? 'tgt' : top === L.src ? 'src' : 'other';
      cardRender(card, { p: Array.from(p), pr, tgt: L.tgt, over: top === L.tgt, names: D.names });
      card.classList.toggle('is-over', top === L.tgt);
      node.querySelectorAll('.mv').forEach((m, k) => m.classList.toggle('used', k >= MOVES - changed.size));
    };
    const showDelta = (d) => {
      clearTimeout(deltaTimer);
      delta.textContent = d > 0 ? `+${d}%` : d < 0 ? `−${-d}%` : '±0%';
      delta.dataset.sign = d > 0 ? 'up' : d < 0 ? 'down' : 'flat';
      delta.classList.remove('is-in'); void delta.offsetWidth; delta.classList.add('is-in');
      deltaTimer = setTimeout(() => delta.classList.remove('is-in'), 1600);
    };
    compute(); render();

    const won = await new Promise((resolve) => {
      const onDown = (e) => {
        const d = e.target.closest('.dt');
        if (!d) return;
        const i = Number(d.dataset.i);
        if (!canClick(i)) {
          if (state === 'play' && changed.has(i)) { d.classList.remove('nope'); void d.offsetWidth; d.classList.add('nope'); }
          return;
        }
        e.preventDefault();
        const before = pr[L.tgt];
        x[i] = 1 - x[i];
        changed.add(i);
        compute(); render();
        showDelta(pr[L.tgt] - before);
        if (argmax(p) === L.tgt) {
          state = 'won';
          node.classList.add('is-won');
          flash(grid, `<b>Zhakowane!</b><span class="flash__sub">AI widzi ${L.tgt} · ${pr[L.tgt]}%</span>`, 'go', 1200);
          action.className = 'il__action is-next';
          action.disabled = false;
          action.textContent = lv === LEVELS - 1 ? 'Zobacz wynik →' : 'Następny poziom →';
          render();
        } else if (changed.size >= MOVES) {
          state = 'lost';
          node.classList.add('is-lost');
          flash(grid, `<b>AI się obroniło</b><span class="flash__sub">Zielone kółka: punkty, które działały</span>`, 'stop', 1600);
          action.className = 'il__action is-lost';
          action.disabled = false;
          action.textContent = lv === LEVELS - 1 ? 'Zobacz wynik →' : 'Następny poziom →';
          render();
        }
      };
      const onAction = () => {
        if (state === 'play') return;
        cleanup(); resolve(state === 'won');
      };
      const cleanup = () => {
        grid.removeEventListener('pointerdown', onDown);
        action.removeEventListener('click', onAction);
      };
      grid.addEventListener('pointerdown', onDown);
      action.addEventListener('click', onAction);
      ctx.signal.addEventListener('abort', () => { cleanup(); resolve(null); }, { once: true });
    });
    if (won == null || ctx.signal.aborted) return null;
    results.push(won);
  }

  const wins = results.filter(Boolean).length;
  ctx.report('illusion', wins / LEVELS, { won: wins });
  const human = wins >= NEED_WINS;
  return {
    score: `${wins}/${LEVELS}`,
    scoreLabel: 'cyfr zhakowanych',
    headline: human ? 'Sieć neuronowa oszukana!' : 'Sieć się obroniła',
    body: el(`<p class="lesson__note lesson__note--big">${human
      ? 'Sieć nie widzi cyfr jak człowiek. Kilka kropek w czułych miejscach zmienia jej decyzję, choć dla nas cyfra wygląda prawie tak samo.'
      : 'Tylko kilka punktów naprawdę zmienia decyzję sieci, reszta to wabiki. Atakujący muszą wiedzieć, gdzie sieć „patrzy”.'}</p>`),
    tone: human ? 'go' : 'stop',
  };
}
