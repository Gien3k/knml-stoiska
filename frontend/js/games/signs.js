// Wektorowe znaki drogowe (viewBox 0 0 100 100) dla gry "Oszukaj Autopilot".
const RED = '#c8102e', WHITE = '#f6f6f2', BLACK = '#141414', YELLOW = '#f5c400', BLUE = '#0052a0', GREEN = '#22c55e';
const font = 'font-family="Barlow Condensed, Barlow, sans-serif" font-weight="700" text-anchor="middle"';
// linia bazowa tak, by środek wielkich liter/cyfr był w środku znaku (wysokość cyfr w Barlow Condensed = 0,706 em)
const ty = (size, cy = 50) => (cy + size * 0.353).toFixed(1);

const oct = (r, cx = 50, cy = 50) => Array.from({ length: 8 }, (_, k) => {
  const a = Math.PI / 8 + (k * Math.PI) / 4;
  return `${(cx + r * Math.cos(a)).toFixed(2)},${(cy + r * Math.sin(a)).toFixed(2)}`;
}).join(' ');

const limit = (n) => `
  <circle cx="50" cy="50" r="48" fill="${RED}"/>
  <circle cx="50" cy="50" r="37" fill="${WHITE}"/>
  <text x="50" y="${ty(n.length > 2 ? 34 : 44)}" ${font} font-size="${n.length > 2 ? 34 : 44}" fill="${BLACK}" letter-spacing="-1">${n}</text>`;

// znak nakazu: niebieskie koło z białą strzałką; strzałka (x 19-81) wyśrodkowana na 50
const ARROW_DIR = { right: '', left: ' transform="translate(100 0) scale(-1 1)"', straight: ' transform="rotate(-90 50 50)"' };
const mandatory = (dir) => `
  <circle cx="50" cy="50" r="48" fill="${BLUE}"/>
  <circle cx="50" cy="50" r="44.5" fill="none" stroke="${WHITE}" stroke-width="2"/>
  <g${ARROW_DIR[dir]} fill="${WHITE}">
    <rect x="19" y="44" width="42" height="12"/>
    <polygon points="55,28 81,50 55,72"/>
  </g>`;

export const SIGNS = {
  stop: `
    <polygon points="${oct(49)}" fill="${WHITE}"/>
    <polygon points="${oct(46)}" fill="${RED}"/>
    <polygon points="${oct(43)}" fill="none" stroke="${WHITE}" stroke-width="1.2"/>
    <text x="50" y="${ty(31)}" ${font} font-size="31" fill="${WHITE}" letter-spacing="1">STOP</text>`,
  limit50: limit('50'),
  limit30: limit('30'),
  limit80: limit('80'),
  limit60: limit('60'),          // tylko podgląd na stronie głównej (nie ma go w grze)
  right: mandatory('right'),
  left: mandatory('left'),
  straight: mandatory('straight'),  // tylko podgląd na stronie głównej
  limit100: limit('100'),
  noentry: `
    <circle cx="50" cy="50" r="48" fill="${RED}"/>
    <rect x="16" y="41" width="68" height="18" fill="${WHITE}"/>`,
  yield: `
    <polygon points="3,8 97,8 50,94" fill="${RED}" stroke="${RED}" stroke-width="4" stroke-linejoin="round"/>
    <polygon points="19,17 81,17 50,74" fill="${YELLOW}"/>`,
  priority: `
    <polygon points="50,2 98,50 50,98 2,50" fill="${BLACK}"/>
    <polygon points="50,5 95,50 50,95 5,50" fill="${WHITE}"/>
    <polygon points="50,20 80,50 50,80 20,50" fill="${YELLOW}"/>`,
  crossing: `
    <rect x="3" y="3" width="94" height="94" rx="6" fill="${BLUE}"/>
    <polygon points="50,11 89,83 11,83" fill="${WHITE}"/>
    <g fill="${BLACK}">
      <rect x="27" y="72" width="9" height="5"/><rect x="40" y="72" width="9" height="5"/><rect x="53" y="72" width="9" height="5"/><rect x="66" y="72" width="9" height="5"/>
      <circle cx="51" cy="36" r="4.2"/>
    </g>
    <g stroke="${BLACK}" stroke-width="4" stroke-linecap="round" fill="none">
      <path d="M50 42 L47 56 M50 44 L58 51 M50 44 L42 50 M47 56 L53 67 M47 56 L40 66"/>
    </g>`,
  parking: `
    <rect x="3" y="3" width="94" height="94" rx="6" fill="${BLUE}"/>
    <rect x="8" y="8" width="84" height="84" rx="3" fill="none" stroke="${WHITE}" stroke-width="2"/>
    <text x="48.9" y="${ty(76)}" ${font} font-size="76" fill="${WHITE}">P</text>`,
};

export const signSvg = (id, cls = '') => `<svg class="${cls}" viewBox="0 0 100 100" aria-hidden="true">${SIGNS[id]}</svg>`;
