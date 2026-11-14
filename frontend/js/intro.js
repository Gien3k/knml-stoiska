// Ekran wstępu przed grą: po lewej podgląd gry na żywo (ta sama animacja co na kafelku strony głównej),
// po prawej pojedynek "TY kontra AI", 3 kluczowe liczby (ostatnia = warunek wygranej), "Zaczynamy" i "Menu".
// Na czas wstępu górny pasek jest schowany (cały opis jest tutaj), więc ekran ma więcej miejsca.
// (Warianty v1, v3-v5 z etapu wyboru zostały usunięte - wybrano podgląd na żywo.)
import { el, esc, icons } from './ui.js';
import { demoMind, demoIllusion, demoAutopilot } from './hub.js';

const DEMOS = { mind: demoMind, illusion: demoIllusion, autopilot: demoAutopilot };

const markup = (g, i) => `
  <div class="intro enter"><div class="in">
    <div class="in-demo panel">
      <span class="in-demo__cap"><i></i>Podgląd gry</span>
      <div class="tile tile--${g.id}"><span class="tile__viz" data-demo></span></div>
    </div>
    <div class="in-text">
      <div class="in-head">
        <span class="i-kicker">${esc(g.tech)}</span>
        <h2 class="i-title">${esc(g.title)}</h2>
      </div>
      <div class="in-vs">
        <div class="in-side in-you"><b>TY</b><p>${esc(i.you)}</p></div>
        <div class="in-side in-ai"><b>AI</b><p>${esc(i.ai)}</p></div>
      </div>
      <div class="in-stats">${i.stats.map(([v, l], k) => `<div class="in-stat${k === i.stats.length - 1 ? ' is-win' : ''}"><b>${esc(v)}</b><span>${esc(l)}</span></div>`).join('')}</div>
      <div class="in-btns">
        <button class="btn btn--primary intro__go" type="button">${icons.play}<span>Zaczynamy</span></button>
        <button class="btn btn--ghost intro__home" type="button">${icons.home}<span>Menu</span></button>
      </div>
    </div>
  </div></div>`;

/** Pokazuje wstęp gry w `stage`; Promise kończy się po kliknięciu "Zaczynamy". */
export function showIntro(stage, game, signal) {
  return new Promise((resolve) => {
    if (!game.intro) { resolve(); return; }
    const node = el(markup(game, game.intro));
    stage.replaceChildren(node);
    let raf = 0;
    const host = node.querySelector('[data-demo]');
    if (DEMOS[game.id]) {
      const demo = DEMOS[game.id](host);
      const t0 = performance.now();
      const loop = (now) => { raf = requestAnimationFrame(loop); if (!document.hidden) demo.frame(now - t0); };
      raf = requestAnimationFrame(loop);
    }
    node.querySelector('.intro__go').addEventListener('click', () => { cancelAnimationFrame(raf); resolve(); }, { once: true });
    node.querySelector('.intro__home').addEventListener('click', () => { location.hash = '#/'; }, { once: true });
    signal.addEventListener('abort', () => cancelAnimationFrame(raf), { once: true });
  });
}
