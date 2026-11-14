// Karta po prawej stronie w grze "Oszukaj Autopilot" - "wyjście sieci":
//  - decyzja autopilota: ikona znaku w ramce detekcji (jak na kamerze) + nazwa + manewr,
//  - jeden pasek 100% podzielony na cel / znak / inne znaki z progiem 80% i legendą z liczbami,
//  - jeden przycisk akcji.
// Liczby (.v-src/.v-tgt/.v-oth) ustawia gra; pasek rysuje się z właściwości CSS --ps/--pt/--po
// (procenty znaku, celu i innych znaków, razem 100), które gra ustawia na karcie.
// (Warianty v2-v4 z etapu wyboru wyglądu zostały usunięte - wybrano ten widok.)
import { esc } from '../ui.js';
import { signSvg } from './signs.js';

const othIcon = '<i class="c-oth" aria-hidden="true">?</i>';

export function cardMarkup(sc, p0) {
  return `<aside class="ap__card ap-card ap-card--1 panel">
    <section class="ap__sec ap__now" data-state="src">
      <span class="ap__lbl">DECYZJA AUTOPILOTA</span>
      <div class="ap__nowrow">
        <span class="c-roi"><span class="ap__nowicon"><span class="is-src">${signSvg(sc.sign)}</span><span class="is-tgt">${signSvg(sc.tgtSign)}</span></span></span>
        <div class="ap__nowtxt">
          <b class="ap__nowname"><span class="is-src">${esc(sc.src.name)}</span><span class="is-tgt">${esc(sc.tgt.name)}</span></b>
          <span class="ap__nowact"><span class="is-src">${esc(sc.src.act)}</span><span class="is-tgt">${esc(sc.tgt.act)}</span></span>
        </div>
      </div>
    </section>
    <section class="ap__sec c1-out">
      <div class="c-head"><span class="ap__lbl">WYJŚCIE SIECI</span><span class="c-sum">razem 100%</span></div>
      <div class="c1-stack"><i class="c1-seg k-tgt"></i><i class="c1-seg k-src"></i><i class="c1-seg k-oth"></i><span class="c1-goal"><em>cel 80%</em></span></div>
      <ul class="c1-legend">
        <li class="k-tgt">${signSvg(sc.tgtSign)}<span>${esc(sc.tgt.name)}</span><b class="v-tgt">${p0[1]}%</b></li>
        <li class="k-src">${signSvg(sc.sign)}<span>${esc(sc.src.name)}</span><b class="v-src">${p0[0]}%</b></li>
        <li class="k-oth">${othIcon}<span>Inne znaki</span><b class="v-oth">${p0[2]}%</b></li>
      </ul>
    </section>
    <section class="ap__sec ap__sec--foot"><button class="ap__btn" type="button"></button></section>
  </aside>`;
}
