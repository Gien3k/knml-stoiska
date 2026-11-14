import { el, esc, icons } from './ui.js';
import { reportResult } from './api.js';
import { showIntro } from './intro.js';

/**
 * Rama gry: górny pasek w JEDNEJ linii (powrót | nazwa gry albo cel rundy | postęp) + scena gry.
 * Gra może ustawić cel rundy przez ctx.setGoal(html) - wtedy na środku paska zamiast nazwy gry
 * pojawia się duży, konkretny cel (bez dodatkowych instrukcji).
 */
export function openGame(view, top, game) {
  const ctrl = new AbortController();
  top.classList.add('top--game');
  top.append(
    el(`<div class="mark"><button class="btn-back" type="button">${icons.back}<span>Powrót do menu</span></button></div>`),
    el(`<div class="top__center"><span class="wordmark">${esc(game.title)}</span></div>`),
    el(`<div class="hud"></div>`),
  );
  const hud = top.querySelector('.hud');
  const center = top.querySelector('.top__center');
  const titleMarkup = center.innerHTML;
  top.querySelector('.btn-back').addEventListener('click', () => { location.hash = '#/'; });

  const stage = el(`<section class="game game--${game.id}"></section>`);
  view.append(stage);

  const ctx = {
    signal: ctrl.signal,
    stage,
    hud,
    setHud(content) {
      if (content instanceof Node) hud.replaceChildren(content);
      else hud.innerHTML = content || '';
    },
    setGoal(html) {
      if (html) { center.innerHTML = `<h1 class="top__goal">${html}</h1>`; top.classList.add('has-goal'); }
      else { center.innerHTML = titleMarkup; top.classList.remove('has-goal'); }
    },
    report: (mode, score, meta) => reportResult(game.id, mode, score, meta),
    home: () => { location.hash = '#/'; },
  };

  let gameCleanup = null;
  const start = () => {
    try {
      gameCleanup = game.module.mount(stage, ctx) || null;
    } catch (e) {
      console.error(e);
      stage.replaceChildren(el(`<div class="note-center">Nie udało się uruchomić gry. Wróć do menu i spróbuj ponownie.</div>`));
    }
  };
  // najpierw krótki wstęp (o czym jest gra i jak wygrać), gra startuje po "Zaczynamy";
  // na czas wstępu górny pasek jest schowany - wstęp ma własny przycisk "Menu"
  const stageEl = top.parentElement;
  stageEl.classList.add('is-intro');
  showIntro(stage, game, ctrl.signal).then(() => {
    stageEl.classList.remove('is-intro');
    if (!ctrl.signal.aborted) start();
  });
  return () => {
    ctrl.abort();
    stageEl.classList.remove('is-intro');
    top.classList.remove('top--game', 'has-goal');
    try { gameCleanup?.(); } catch (e) { console.error(e); }
  };
}
