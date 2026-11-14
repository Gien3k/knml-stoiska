// Gra 1: "Przechytrz algorytm" - pojedynek z predyktorem Markowa (n-gramy 1-4, mieszanka bayesowska).
// Na ekranie tylko wynik pojedynku; cała matematyka działa pod spodem.
import { el, resultScreen, wait, pct } from '../ui.js';
import { MarkovPredictor, switchRate, trigramStats } from '../ml/markov.js';
import { getCrowdNgram, postSequence } from '../api.js';

const WARM = 8;   // rozgrzewka: AI tylko obserwuje i uczy się stylu gracza
const N = 20;      // właściwy pojedynek

export function mount(stage, ctx) {
  (async () => {
    while (!ctx.signal.aborted) {
      const res = await play(stage, ctx);
      if (!res || ctx.signal.aborted) return;
      ctx.setHud('');
      const act = await resultScreen(stage, res);
      if (act === 'home') return ctx.home();
    }
  })();
}

/** Jedno zdanie wniosku - najwyraźniejszy ludzki nawyk w ciągu gracza. */
function conclusion(seq, acc) {
  const sw = switchRate(seq);
  const tri = Object.fromEntries(trigramStats(seq).map((t) => [t.k, t.ratio]));
  const ones = seq.reduce((a, b) => a + b, 0) / seq.length;
  if (sw > 0.62) return 'Zbyt często zmieniasz cyfrę. Ludzie robią to podświadomie!';
  if (tri['000'] + tri['111'] < 0.5) return 'Unikasz długich serii, a prawdziwy los tworzy je bez skrupułów!';
  if (sw < 0.38) return 'Za rzadko zmieniasz cyfrę, a długie serie łatwo przewidzieć!';
  if (ones > 0.64) return 'Wybierasz jedynkę częściej niż zero. AI szybko to zauważyło!';
  if (ones < 0.36) return 'Wybierasz zero częściej niż jedynkę. AI szybko to zauważyło!';
  if (acc <= 0.5) return 'Twoje kliknięcia były naprawdę trudne do przewidzenia. Gratulacje!';
  return 'AI wyłapało powtarzające się wzorce w twoich kliknięciach.';
}

async function play(stage, ctx) {
  const pred = new MarkovPredictor(await getCrowdNgram());
  if (ctx.signal.aborted) return null;
  const seq = [];
  const warmSeq = [];
  let hits = 0;
  let phase = 'warm';   // warm | switch | duel

  const node = el(`
    <div class="mk mk--simple enter" data-phase="warm">
      <div class="warm">
        <h2 class="warm__t">Najpierw pokaż AI, jak klikasz</h2>
        <div class="warm__bits">${Array.from({ length: WARM }, () => '<span class="wslot"></span>').join('')}</div>
        <p class="warm__s">AI obserwuje twoje pierwsze ${WARM} kliknięć i uczy się, jak klikasz</p>
      </div>
      <div class="duel">
        <div class="duel__score">
          <span class="duel__side duel__side--you"><small>TY</small><b class="d-you">0</b></span>
          <span class="duel__sep">:</span>
          <span class="duel__side duel__side--ai"><b class="d-ai">0</b><small>AI</small></span>
        </div>
        <div class="duel__line">Przewidywalność twojego mózgu: <b class="d-pct">0%</b></div>
        <div class="duel__bar"><i class="d-bar"></i><span class="duel__half"></span></div>
      </div>
      <div class="mk__strip">${Array.from({ length: N }, (_, i) => `<span class="slot" data-i="${i}"></span>`).join('')}<b class="mk__count"><span class="mk-c">0</span>/${N}</b></div>
      <div class="mk__keys">
        <button class="key" data-b="0" aria-label="Zero">0</button>
        <div class="bet" aria-live="polite" data-state="learn"><span class="bet__lbl">AI obserwuje</span><b class="bet__v">…</b><svg class="bet__mkg" viewBox="0 0 300 130" aria-hidden="true"><defs><marker id="mkgArrow" viewBox="0 0 10 10" refX="8" refY="5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#6cb4ff"/></marker></defs><path class="mkg-e mkg-e01" d="M108 46 Q150 8 192 46" marker-end="url(#mkgArrow)"/><path class="mkg-e mkg-e10" d="M192 84 Q150 122 108 84" marker-end="url(#mkgArrow)"/><path class="mkg-e mkg-e00" d="M72 46 C22 8 22 122 72 84" marker-end="url(#mkgArrow)"/><path class="mkg-e mkg-e11" d="M228 46 C278 8 278 122 228 84" marker-end="url(#mkgArrow)"/><g class="mkg-n mkg-n0"><circle cx="90" cy="65" r="26"/><text x="90" y="75.6">0</text></g><g class="mkg-n mkg-n1"><circle cx="210" cy="65" r="26"/><text x="210" y="75.6">1</text></g></svg><span class="bet__msg">uczy się, jak klikasz</span></div>
        <button class="key" data-b="1" aria-label="Jeden">1</button>
      </div>
    </div>`);
  stage.replaceChildren(node);
  const $ = (s) => node.querySelector(s);
  const slots = [...node.querySelectorAll('.slot')];
  const keys = [...node.querySelectorAll('.key')];
  const wslots = [...node.querySelectorAll('.wslot')];
  // łańcuch Markowa na żywo: grubość strzałki = jak często po danej cyfrze wybierasz następną
  const trans = { '00': 0, '01': 0, '10': 0, '11': 0 };
  let prevBit = null;
  const pulse = (el) => { el.classList.remove('hit'); void el.getBoundingClientRect(); el.classList.add('hit'); };
  const markov = (b) => {
    if (prevBit != null) trans[`${prevBit}${b}`] += 1;
    const tot = Object.values(trans).reduce((a, c) => a + c, 0);
    for (const [k, c] of Object.entries(trans)) {
      const e = node.querySelector(`.mkg-e${k}`);
      const share = tot ? c / tot : 0;
      e.style.strokeWidth = (2 + 10 * share).toFixed(1);
      e.style.opacity = (0.2 + 0.8 * share).toFixed(2);
      if (prevBit != null && k === `${prevBit}${b}`) pulse(e);
    }
    pulse(node.querySelector(`.mkg-n${b}`));
    prevBit = b;
  };
  wslots[0].classList.add('cur');

  let bet = null;
  let revealT = 0;
  const render = () => {
    const n = seq.length;
    const acc = n ? hits / n : 0;
    $('.d-you').textContent = String(n - hits);
    $('.d-ai').textContent = String(hits);
    $('.d-pct').textContent = pct(acc);
    $('.d-bar').style.width = `${acc * 100}%`;
    node.dataset.lead = !n ? '' : acc > 0.5 ? 'ai' : acc < 0.5 ? 'you' : 'tie';
    $('.mk-c').textContent = String(n);
  };
  render();

  const done = new Promise((resolve) => {
    const press = (b) => {
      if (ctx.signal.aborted || phase === 'switch') return;
      keys[b].classList.remove('pressed'); void keys[b].offsetWidth; keys[b].classList.add('pressed');
      if (phase === 'warm') {
        // rozgrzewka: AI tylko się uczy, nic nie obstawia; rysunek pokazuje, czego się uczy (przejścia 0/1)
        pred.update(b);
        markov(b);
        warmSeq.push(b);
        const w = warmSeq.length - 1;
        wslots[w].textContent = String(b);
        wslots[w].classList.remove('cur');
        wslots[w].classList.add('done');
        wslots[w + 1]?.classList.add('cur');
        if (warmSeq.length >= WARM) {
          phase = 'switch';
          $('.warm__t').textContent = 'Gotowe! Teraz AI zgaduje twój ruch';
          node.classList.add('is-switch');
          setTimeout(() => {
            if (ctx.signal.aborted) return;
            phase = 'duel';
            node.dataset.phase = 'duel';
            node.classList.remove('is-switch');
            slots[0].classList.add('cur');
            bet = pred.predict();
            const betEl = $('.bet');
            betEl.dataset.state = '';
            $('.bet__lbl').textContent = 'AI już obstawiło';
            $('.bet__v').textContent = '?';
            $('.bet__msg').textContent = '';
          }, 1300);
        }
        return;
      }
      if (seq.length >= N) return;
      const hit = bet.bit === b;
      pred.update(b);
      seq.push(b);
      if (hit) hits++;
      const i = seq.length - 1;
      slots[i].textContent = String(b);
      slots[i].classList.remove('cur');
      slots[i].classList.add(hit ? 'ai' : 'you');
      slots[i + 1]?.classList.add('cur');
      node.classList.remove('flash-ai', 'flash-you'); void node.offsetWidth;
      node.classList.add(hit ? 'flash-ai' : 'flash-you');
      const betEl = $('.bet');
      betEl.dataset.state = hit ? 'ai' : 'you';
      $('.bet__lbl').textContent = 'AI obstawiło';
      $('.bet__v').textContent = String(bet.bit);
      $('.bet__msg').textContent = hit ? 'AI TRAFIŁO!' : 'AI OSZUKANE!';
      clearTimeout(revealT);
      revealT = setTimeout(() => {
        betEl.dataset.state = '';
        $('.bet__lbl').textContent = 'AI już obstawiło';
        $('.bet__v').textContent = seq.length < N ? '?' : '';
        $('.bet__msg').textContent = '';
      }, 800);
      if (seq.length < N) bet = pred.predict();
      render();
      if (seq.length >= N) resolve(true);
    };
    node.querySelector('.mk__keys').addEventListener('pointerdown', (e) => {
      const k = e.target.closest('.key');
      if (k) { e.preventDefault(); press(Number(k.dataset.b)); }
    });
    window.addEventListener('keydown', (e) => {
      if (e.repeat) return;
      if (e.key === '0' || e.key === 'ArrowLeft') press(0);
      if (e.key === '1' || e.key === 'ArrowRight') press(1);
    }, { signal: ctx.signal });
    ctx.signal.addEventListener('abort', () => resolve(false), { once: true });
  });
  if (!(await done)) return null;
  if (!(await wait(1000, ctx.signal))) return null;
  clearTimeout(revealT);

  const acc = hits / N;
  postSequence(warmSeq.concat(seq).join(''));
  ctx.report('mind', 1 - acc, { acc, seq: seq.join('') });
  const headline = acc > 0.5 ? `AI przejrzało cię w ${pct(acc)}` : acc < 0.5 ? `AI przejrzało cię tylko w ${pct(acc)}` : 'Idealny remis: 50%';
  return {
    score: `${N - hits} : ${hits}`,
    scoreLabel: 'Ty : AI',
    headline,
    body: el(`<p class="lesson__note lesson__note--big">${conclusion(seq, acc)}</p>`),
    tone: acc > 0.5 ? 'stop' : acc < 0.5 ? 'go' : 'idle',
  };
}
