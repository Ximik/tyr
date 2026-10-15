const shapeCases = {
  valid: {
    source: `def project {device : Device}
    (x : Tensor { shape := #[32, 768], dtype := .Float32, device := device })
    (weight : Tensor { shape := #[512, 768], dtype := .Float32, device := device }) :
    Tensor { shape := #[32, 512], dtype := .Float32, device := device } :=
  linear x weight`,
    result: `project {device : Device}
  (x : Tensor { shape := #[32, 768], dtype := Float32, device := device })
  (weight : Tensor { shape := #[512, 768], dtype := Float32, device := device }) :
  Tensor { shape := #[32, 512], dtype := Float32, device := device }

Accepted during elaboration.`,
    className: 'ok-line'
  },
  invalid: {
    source: `def brokenProject {device : Device}
    (x : Tensor { shape := #[32, 768], dtype := .Float32, device := device })
    (weight : Tensor { shape := #[768, 512], dtype := .Float32, device := device }) :
    Tensor { shape := #[32, 768], dtype := .Float32, device := device } :=
  linear x weight`,
    result: `error: Application type mismatch

The argument weight has type
  Tensor { shape := #[768, 512], dtype := Float32, device := device }
but is expected to have type
  Tensor { shape := #[768, 768], dtype := Float32, device := device }
in the application
  x.linear weight

Rejected during elaboration.`,
    className: 'error-line'
  }
};

const sourceCode = document.querySelector('#shape-source code');
const resultCode = document.querySelector('#shape-result code');

function selectShapeCase(name) {
  const selected = shapeCases[name];
  sourceCode.textContent = selected.source;
  resultCode.textContent = selected.result;
  resultCode.className = selected.className;
  for (const button of document.querySelectorAll('[data-shape-case]')) {
    const active = button.dataset.shapeCase === name;
    button.classList.toggle('selected', active);
    button.setAttribute('aria-pressed', String(active));
  }
}

for (const button of document.querySelectorAll('[data-shape-case]')) {
  button.addEventListener('click', () => selectShapeCase(button.dataset.shapeCase));
}
selectShapeCase('valid');

const NS = 'http://www.w3.org/2000/svg';
const contact = window.TYR_CONTACT_DATA;

function svgElement(name, attributes = {}, content = '') {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (content) node.textContent = content;
  return node;
}

function appendLine(svg, x1, y1, x2, y2, attributes = {}) {
  svg.append(svgElement('line', {x1, y1, x2, y2, ...attributes}));
}

function appendText(svg, x, y, value, attributes = {}) {
  svg.append(svgElement('text', {x, y, ...attributes}, value));
}

function buildContactChart() {
  if (!contact) return;
  const svg = document.getElementById('contact-chart');
  const samples = contact.samples;
  const x0 = 72, x1 = 866;
  const panels = [
    {key: 'position', y0: 40, y1: 210, min: 0, max: .48, label: 'position'},
    {key: 'velocity', y0: 285, y1: 455, min: -3.2, max: 1.4, label: 'velocity'}
  ];
  const sx = time => x0 + time / .4 * (x1 - x0);
  const sy = (value, panel) => panel.y1 - (value - panel.min) / (panel.max - panel.min) * (panel.y1 - panel.y0);

  for (const panel of panels) {
    for (let i = 0; i <= 4; i++) {
      const y = panel.y0 + i / 4 * (panel.y1 - panel.y0);
      const value = panel.max - i / 4 * (panel.max - panel.min);
      appendLine(svg, x0, y, x1, y, {stroke: '#e4e4df', 'stroke-width': 1});
      appendText(svg, x0 - 10, y + 4, value.toFixed(1), {'text-anchor': 'end', fill: '#6d737b', 'font-size': 12});
    }
    appendLine(svg, x0, panel.y0, x0, panel.y1, {stroke: '#545b65', 'stroke-width': 1.2});
    appendLine(svg, x0, panel.y1, x1, panel.y1, {stroke: '#545b65', 'stroke-width': 1.2});
    appendText(svg, 18, (panel.y0 + panel.y1) / 2, panel.label, {fill: '#424850', 'font-size': 13, transform: `rotate(-90 18 ${(panel.y0 + panel.y1) / 2})`, 'text-anchor': 'middle'});
    const points = samples.map(sample => `${sx(sample.time).toFixed(2)},${sy(sample[panel.key], panel).toFixed(2)}`).join(' ');
    svg.append(svgElement('polyline', {points, fill: 'none', stroke: panel.key === 'position' ? '#3974a4' : '#586b83', 'stroke-width': 2.6, 'stroke-linejoin': 'round'}));
  }
  for (let i = 0; i <= 4; i++) {
    const time = i / 10;
    const x = sx(time);
    appendText(svg, x, 483, time.toFixed(1), {'text-anchor': 'middle', fill: '#6d737b', 'font-size': 12});
  }
  appendText(svg, (x0 + x1) / 2, 499, 'time', {'text-anchor': 'middle', fill: '#424850', 'font-size': 13});
  const impactX = sx(contact.impact.time);
  appendLine(svg, impactX, panels[0].y0, impactX, panels[1].y1, {stroke: '#b85a3c', 'stroke-width': 1.5, 'stroke-dasharray': '5 5'});
  appendText(svg, impactX + 7, 55, 'impact, τ = 0.2', {fill: '#a04b33', 'font-size': 12});
  for (const panel of panels) {
    const marker = svgElement('circle', {r: 5, fill: '#fff', stroke: '#b85a3c', 'stroke-width': 2});
    marker.id = `contact-marker-${panel.key}`;
    svg.append(marker);
  }
}

function updateContactState(index) {
  const sample = contact.samples[index];
  const chart = document.getElementById('contact-chart');
  const x = 72 + sample.time / .4 * (866 - 72);
  const positionY = 210 - (sample.position - 0) / .48 * (210 - 40);
  const velocityY = 455 - (sample.velocity + 3.2) / 4.6 * (455 - 285);
  const positionMarker = chart.querySelector('#contact-marker-position');
  const velocityMarker = chart.querySelector('#contact-marker-velocity');
  positionMarker.setAttribute('cx', x); positionMarker.setAttribute('cy', positionY);
  velocityMarker.setAttribute('cx', x); velocityMarker.setAttribute('cy', velocityY);

  document.getElementById('contact-time').value = `t = ${sample.time.toFixed(3)}`;
  document.getElementById('contact-phase').textContent = sample.phase === 'pre' ? 'pre-impact' : 'post-impact';
  document.getElementById('contact-position').textContent = sample.position.toFixed(4);
  document.getElementById('contact-velocity').textContent = sample.velocity.toFixed(4).replace('-', '−');

  const view = document.getElementById('probe-view');
  view.replaceChildren();
  const floorY = 252;
  appendLine(view, 35, floorY, 285, floorY, {stroke: '#555d67', 'stroke-width': 3});
  for (let stripe = 0; stripe < 11; stripe++) appendLine(view, 45 + stripe * 23, floorY, 28 + stripe * 23, floorY + 15, {stroke: '#c7c9c6', 'stroke-width': 1});
  const centerY = floorY - 24 - Math.max(sample.position - .05, 0) / .43 * 165;
  view.append(svgElement('circle', {cx: 160, cy: centerY, r: 24, fill: '#e7eef4', stroke: '#3974a4', 'stroke-width': 2.5}));
  const arrowScale = 22;
  const arrowEnd = centerY - sample.velocity * arrowScale;
  appendLine(view, 205, centerY, 205, arrowEnd, {stroke: sample.velocity < 0 ? '#b85a3c' : '#3974a4', 'stroke-width': 3});
  const direction = arrowEnd < centerY ? -1 : 1;
  view.append(svgElement('path', {d: `M 198 ${arrowEnd - direction * 9} L 205 ${arrowEnd} L 212 ${arrowEnd - direction * 9}`, fill: 'none', stroke: sample.velocity < 0 ? '#b85a3c' : '#3974a4', 'stroke-width': 3}));
  appendText(view, 160, 282, 'contact plane', {'text-anchor': 'middle', fill: '#6d737b', 'font-size': 12});
}

if (contact) {
  buildContactChart();
  const slider = document.getElementById('contact-step');
  slider.max = String(contact.samples.length - 1);
  slider.addEventListener('input', () => updateContactState(Number(slider.value)));
  updateContactState(0);
}
