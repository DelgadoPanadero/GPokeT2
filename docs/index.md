# GPokeT2 — Pokémon Sprite Generator

![gpoket2](gpoket2.png)

A GPT-2 based autoregressive model that generates 64×64 Pokémon sprites token by token,
conditioned on type, generation, evolution stage and more.

| Pokemon sprite | | ASCII representation | | Train the model |
|:---------------:|:--:|:--------------------:|:--:|:--:|
| <img src="sprite_image.png" width="160"/> | → | <img src="sprite_ascii.png" width="160"/> | → | GPT2-Small |

---

# Pokédex

<style>
/* ── Pokédex layout ───────────────────────────────────────── */
.pdx-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  background: var(--md-primary-fg-color);
  color: #fff;
  padding: 1rem 1.5rem;
  border-radius: 12px;
  margin-bottom: 1.5rem;
}
.pdx-header h2 { margin: 0; font-size: 1.4rem; letter-spacing: .05em; }
.pdx-count {
  margin-left: auto;
  background: rgba(255,255,255,.2);
  padding: .25rem .75rem;
  border-radius: 20px;
  font-size: .85rem;
}

/* ── Filter bar ───────────────────────────────────────────── */
.pdx-filters {
  display: flex;
  flex-wrap: wrap;
  gap: .5rem;
  margin-bottom: 1.5rem;
}
.pdx-filter-btn {
  padding: .3rem .85rem;
  border: 2px solid transparent;
  border-radius: 20px;
  font-size: .78rem;
  font-weight: 600;
  cursor: pointer;
  text-transform: capitalize;
  transition: opacity .15s, transform .1s;
  color: #fff;
}
.pdx-filter-btn:hover { opacity: .85; transform: scale(1.05); }
.pdx-filter-btn.active { border-color: #fff; box-shadow: 0 0 0 2px currentColor; }

/* ── Grid ─────────────────────────────────────────────────── */
.pdx-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 1rem;
}

/* ── Card ─────────────────────────────────────────────────── */
.pdx-card {
  background: var(--md-code-bg-color);
  border: 2px solid var(--md-primary-fg-color--light, #5c6bc0);
  border-radius: 12px;
  padding: .75rem .5rem .6rem;
  text-align: center;
  cursor: pointer;
  transition: transform .15s, box-shadow .15s;
  position: relative;
  overflow: hidden;
}
.pdx-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 6px 20px rgba(63,81,181,.35);
}
.pdx-number {
  font-size: .7rem;
  font-weight: 700;
  color: var(--md-primary-fg-color);
  letter-spacing: .08em;
  margin-bottom: .4rem;
}
/* Pokédex screen */
.pdx-screen {
  width: 104px;
  height: 104px;
  margin: 0 auto .6rem;
  background: #1a1a2e;
  border-radius: 14px;
  border: 3px solid var(--md-primary-fg-color);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: inset 0 0 12px rgba(63,81,181,.5);
  overflow: hidden;
}
.pdx-screen img {
  width: 72px;
  height: 72px;
  image-rendering: pixelated;
  image-rendering: crisp-edges;
}
.pdx-types {
  display: flex;
  justify-content: center;
  gap: .3rem;
  flex-wrap: wrap;
}

/* ── Type badge ───────────────────────────────────────────── */
.type-badge {
  font-size: .65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .06em;
  padding: .15rem .5rem;
  border-radius: 10px;
  color: #fff;
}

/* ── Modal overlay ────────────────────────────────────────── */
.pdx-modal-overlay {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,.7);
  z-index: 9999;
  align-items: center;
  justify-content: center;
}
.pdx-modal-overlay.open { display: flex; }

.pdx-modal {
  background: var(--md-default-bg-color);
  border: 3px solid var(--md-primary-fg-color);
  border-radius: 20px;
  width: min(440px, 92vw);
  overflow: hidden;
  box-shadow: 0 20px 60px rgba(0,0,0,.5);
  animation: pdx-pop .18s ease-out;
}
@keyframes pdx-pop {
  from { transform: scale(.85); opacity: 0; }
  to   { transform: scale(1);   opacity: 1; }
}

/* Modal top bar */
.pdx-modal-bar {
  background: var(--md-primary-fg-color);
  padding: .75rem 1rem;
  display: flex;
  align-items: center;
  gap: .6rem;
}
.pdx-modal-bar .pdx-dot {
  width: 12px; height: 12px; border-radius: 50%; background: #ef5350;
}
.pdx-modal-bar .pdx-dot.y { background: #fdd835; }
.pdx-modal-bar .pdx-dot.g { background: #66bb6a; }
.pdx-modal-title {
  color: #fff;
  font-weight: 700;
  font-size: 1rem;
  letter-spacing: .05em;
  margin-left: .5rem;
}
.pdx-modal-close {
  margin-left: auto;
  background: rgba(255,255,255,.2);
  border: none;
  color: #fff;
  font-size: 1.1rem;
  width: 28px; height: 28px;
  border-radius: 50%;
  cursor: pointer;
  display: flex; align-items: center; justify-content: center;
}
.pdx-modal-close:hover { background: rgba(255,255,255,.35); }

/* Modal body */
.pdx-modal-body {
  padding: 1.5rem;
}
.pdx-modal-screen {
  width: 180px;
  height: 180px;
  background: #1a1a2e;
  border-radius: 20px;
  border: 4px solid var(--md-primary-fg-color);
  margin: 0 auto 1.25rem;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: inset 0 0 24px rgba(63,81,181,.6), 0 4px 16px rgba(63,81,181,.3);
  overflow: hidden;
}
.pdx-modal-screen img {
  width: 128px;
  height: 128px;
  image-rendering: pixelated;
  image-rendering: crisp-edges;
}
.pdx-modal-types {
  display: flex;
  justify-content: center;
  gap: .5rem;
  margin-bottom: 1.25rem;
}
.pdx-modal-types .type-badge { font-size: .8rem; padding: .25rem .75rem; }

.pdx-stats {
  border: 1px solid var(--md-primary-fg-color--light, #5c6bc0);
  border-radius: 10px;
  overflow: hidden;
}
.pdx-stat-row {
  display: flex;
  padding: .5rem .9rem;
  font-size: .85rem;
  border-bottom: 1px solid var(--md-code-bg-color);
}
.pdx-stat-row:last-child { border-bottom: none; }
.pdx-stat-row:nth-child(even) { background: var(--md-code-bg-color); }
.pdx-stat-label {
  color: var(--md-primary-fg-color);
  font-weight: 600;
  width: 140px;
  flex-shrink: 0;
}
</style>

<div class="pdx-header">
  <svg width="28" height="28" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/>
  </svg>
  <h2>GPokeT2 Pokédex</h2>
  <span class="pdx-count" id="pdx-count">40 entries</span>
</div>

<div class="pdx-filters" id="pdx-filters"></div>

<div class="pdx-grid" id="pdx-grid"></div>

<!-- Modal -->
<div class="pdx-modal-overlay" id="pdx-overlay">
  <div class="pdx-modal">
    <div class="pdx-modal-bar">
      <div class="pdx-dot"></div>
      <div class="pdx-dot y"></div>
      <div class="pdx-dot g"></div>
      <span class="pdx-modal-title" id="pdx-modal-title">—</span>
      <button class="pdx-modal-close" id="pdx-modal-close">✕</button>
    </div>
    <div class="pdx-modal-body">
      <div class="pdx-modal-screen">
        <img id="pdx-modal-img" src="" alt="">
      </div>
      <div class="pdx-modal-types" id="pdx-modal-types"></div>
      <div class="pdx-stats" id="pdx-modal-stats"></div>
    </div>
  </div>
</div>

<script>
(function () {

const TYPE_COLORS = {
  normal:   '#A8A878', fire:     '#F08030', water:    '#6890F0',
  electric: '#F8D030', grass:    '#78C850', ice:      '#98D8D8',
  fighting: '#C03028', poison:   '#A040A0', ground:   '#E0C068',
  flying:   '#A890F0', psychic:  '#F85888', bug:      '#A8B820',
  rock:     '#B8A038', ghost:    '#705898', dragon:   '#7038F8',
  dark:     '#705848', steel:    '#B8B8D0', fairy:    '#EE99AC',
  unk:      '#888888',
};

const FILES = [
  "pokemon_bug-bug_sh0_g4_ev1_he0.png",
  "pokemon_dark-flying_sh0_g3_ev3_he0.png",
  "pokemon_dark-grass_sh0_g3_ev0_he1_ryled9.png",
  "pokemon_dark-ground_sh0_g3_ev2_he1_wzseas.png",
  "pokemon_dragon-fighting_sh0_g4_ev0_he1.png",
  "pokemon_dragon-fire_sh0_g3_ev1_he0_cfd3ze.png",
  "pokemon_electric-flying_sh0_g4_ev1_he0.png",
  "pokemon_electric-ice_sh0_g3_ev0_he1_0e56hv.png",
  "pokemon_electric-steel_sh0_g4_ev3_he0.png",
  "pokemon_fairy-steel_sh0_g4_ev1_he1.png",
  "pokemon_fighting-dark_sh0_g3_ev2_he0.png",
  "pokemon_fighting-dragon_sh0_g4_ev2_he0.png",
  "pokemon_fighting-fire_sh0_g3_ev2_he0_6j9xtm.png",
  "pokemon_fire-normal_sh0_g4_ev0_he1.png",
  "pokemon_flying_sh0_g4_ev2_he1.png",
  "pokemon_ghost-dragon_sh0_g4_ev0_he1.png",
  "pokemon_ghost-electric_sh0_g4_ev0_he0_v2gqex.png",
  "pokemon_grass-fire_sh0_g3_ev3_he1_m5k5ac.png",
  "pokemon_grass-ghost_sh0_g4_ev1_he1_1aklfk.png",
  "pokemon_grass-ground_sh0_g3_ev3_he0_p6jj7l.png",
  "pokemon_grass-ice_sh0_g4_ev3_he0.png",
  "pokemon_ground-bug_sh0_g3_ev3_he0_rhd4yu.png",
  "pokemon_ground-steel_sh0_g4_ev3_he0.png",
  "pokemon_ground_sh0_g4_ev0_he0_w0jctb.png",
  "pokemon_ice-fire_sh0_g4_ev2_he0_82iflx.png",
  "pokemon_ice-psychic_sh0_g4_ev0_he0.png",
  "pokemon_normal-dark_sh0_g4_ev2_he0.png",
  "pokemon_normal-poison_sh0_g3_ev0_he1.png",
  "pokemon_psychic-flying_sh0_g3_ev3_he0_plzgr7.png",
  "pokemon_rock-ghost_sh0_g3_ev1_he0_7yepvr.png",
  "pokemon_rock-grass_sh0_g4_ev0_he1.png",
  "pokemon_rock-ground_sh0_g4_ev0_he1.png",
  "pokemon_rock_sh0_g3_ev3_he0_dbznxc.png",
  "pokemon_rock_sh0_g4_ev2_he0.png",
  "pokemon_steel-dragon_sh0_g4_ev1_he0.png",
  "pokemon_steel-fairy_sh0_g3_ev3_he0.png",
  "pokemon_steel-fire_sh0_g3_ev0_he1.png",
  "pokemon_unk-fire_sh0_g4_ev1_he0_ahm7fg.png",
  "pokemon_water-bug_sh0_g3_ev1_he1_jez4il.png",
  "pokemon_water-ground_sh0_g3_ev3_he1.png",
];

const IMG_BASE = 'pokemons/';

function parse(filename) {
  const base = filename.replace(/^pokemon_/, '').replace(/\.png$/, '');
  const parts = base.split('_');
  const types = parts[0].split('-');
  const type1 = types[0];
  const type2 = types[1] || null;
  let shiny = false, gen = null, ev = null, hasEvo = null;
  for (let i = 1; i < parts.length; i++) {
    const p = parts[i];
    if (p.startsWith('sh'))  shiny  = p === 'sh1';
    else if (p.startsWith('g') && /^g\d+$/.test(p)) gen = parseInt(p.slice(1));
    else if (p.startsWith('ev')) ev  = parseInt(p.slice(2));
    else if (p.startsWith('he')) hasEvo = p === 'he1';
  }
  return { filename, type1, type2, shiny, gen, ev, hasEvo };
}

function badge(type) {
  const color = TYPE_COLORS[type] || '#888';
  return `<span class="type-badge" style="background:${color}">${type}</span>`;
}

const data = FILES.map((f, i) => ({ ...parse(f), num: i + 1 }));

/* ── Filter buttons ── */
const allTypes = [...new Set(data.flatMap(p => [p.type1, p.type2].filter(Boolean)))].sort();
let activeType = null;

function renderFilters() {
  const wrap = document.getElementById('pdx-filters');
  const allBtn = makeFilterBtn('All', null, '#3f51b5');
  wrap.appendChild(allBtn);
  allTypes.forEach(t => wrap.appendChild(makeFilterBtn(t, t, TYPE_COLORS[t] || '#888')));
  updateFilterActive();
}

function makeFilterBtn(label, type, color) {
  const btn = document.createElement('button');
  btn.className = 'pdx-filter-btn';
  btn.textContent = label;
  btn.style.background = color;
  btn.dataset.type = type || '';
  btn.addEventListener('click', () => {
    activeType = type;
    updateFilterActive();
    renderGrid();
  });
  return btn;
}

function updateFilterActive() {
  document.querySelectorAll('.pdx-filter-btn').forEach(b => {
    b.classList.toggle('active', (b.dataset.type || null) === activeType);
  });
}

/* ── Grid ── */
function renderGrid() {
  const grid = document.getElementById('pdx-grid');
  const count = document.getElementById('pdx-count');
  const filtered = activeType
    ? data.filter(p => p.type1 === activeType || p.type2 === activeType)
    : data;
  count.textContent = `${filtered.length} entries`;
  grid.innerHTML = '';
  filtered.forEach(p => {
    const card = document.createElement('div');
    card.className = 'pdx-card';
    card.innerHTML = `
      <div class="pdx-number">#${String(p.num).padStart(3,'0')}</div>
      <div class="pdx-screen">
        <img src="${IMG_BASE}${p.filename}" alt="${p.type1}${p.type2 ? '/'+p.type2 : ''}">
      </div>
      <div class="pdx-types">
        ${badge(p.type1)}${p.type2 ? badge(p.type2) : ''}
      </div>`;
    card.addEventListener('click', () => openModal(p));
    grid.appendChild(card);
  });
}

/* ── Modal ── */
const overlay  = document.getElementById('pdx-overlay');
const modalImg = document.getElementById('pdx-modal-img');
const modalTitle = document.getElementById('pdx-modal-title');
const modalTypes = document.getElementById('pdx-modal-types');
const modalStats = document.getElementById('pdx-modal-stats');

function openModal(p) {
  modalTitle.textContent = `#${String(p.num).padStart(3,'0')} · ${p.type1}${p.type2 ? ' / '+p.type2 : ''}`;
  modalImg.src = `${IMG_BASE}${p.filename}`;
  modalImg.alt = modalTitle.textContent;
  modalTypes.innerHTML = badge(p.type1) + (p.type2 ? badge(p.type2) : '');
  const evLabels = ['Basic', 'Stage 1', 'Stage 2', 'Other'];
  modalStats.innerHTML = `
    <div class="pdx-stat-row"><span class="pdx-stat-label">Primary type</span>${p.type1}</div>
    <div class="pdx-stat-row"><span class="pdx-stat-label">Secondary type</span>${p.type2 || '—'}</div>
    <div class="pdx-stat-row"><span class="pdx-stat-label">Generation</span>Gen ${p.gen}</div>
    <div class="pdx-stat-row"><span class="pdx-stat-label">Evolution stage</span>${evLabels[p.ev] ?? p.ev}</div>
    <div class="pdx-stat-row"><span class="pdx-stat-label">Has evolution</span>${p.hasEvo ? 'Yes' : 'No'}</div>
    <div class="pdx-stat-row"><span class="pdx-stat-label">Shiny</span>${p.shiny ? '✨ Yes' : 'No'}</div>`;
  overlay.classList.add('open');
}

document.getElementById('pdx-modal-close').addEventListener('click', () => overlay.classList.remove('open'));
overlay.addEventListener('click', e => { if (e.target === overlay) overlay.classList.remove('open'); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') overlay.classList.remove('open'); });

renderFilters();
renderGrid();

})();
</script>
