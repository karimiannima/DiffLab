/**
 * 2D toy distributions for DiffLab (normalized roughly to [-1.8, 1.8]).
 */

function randn() {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function addNoise(pts, sigma = 0.05) {
  return pts.map(([x, y]) => [x + randn() * sigma, y + randn() * sigma]);
}

export function sampleRings(n = 512) {
  const pts = [];
  const rings = [
    { r: 0.55, w: 0.04 },
    { r: 1.15, w: 0.05 },
    { r: 1.7, w: 0.05 },
  ];
  for (let i = 0; i < n; i++) {
    const ring = rings[i % rings.length];
    const theta = Math.random() * Math.PI * 2;
    const r = ring.r + randn() * ring.w;
    pts.push([r * Math.cos(theta), r * Math.sin(theta)]);
  }
  return pts;
}

export function sampleMoons(n = 512) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const upper = i % 2 === 0;
    const t = Math.random() * Math.PI;
    let x = Math.cos(t);
    let y = Math.sin(t);
    if (!upper) {
      x = 1 - x;
      y = -y - 0.35;
    } else {
      y = y - 0.15;
    }
    x = x * 1.1 - 0.55;
    y = y * 1.1;
    pts.push([x + randn() * 0.06, y + randn() * 0.06]);
  }
  return pts;
}

export function sampleSpiral(n = 512) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const arm = i % 2;
    const t = (i / n) * 3.2 * Math.PI + arm * Math.PI;
    const r = 0.25 + (t / (3.2 * Math.PI)) * 1.5;
    const x = r * Math.cos(t + arm);
    const y = r * Math.sin(t + arm);
    pts.push([x + randn() * 0.05, y + randn() * 0.05]);
  }
  return pts;
}

export function sampleSwissRoll(n = 512) {
  // 2D projection of swiss-roll-like spiral band
  const pts = [];
  for (let i = 0; i < n; i++) {
    const t = 1.5 * Math.PI * (1 + 2 * Math.random());
    const x = t * Math.cos(t) / 5.5;
    const y = t * Math.sin(t) / 5.5;
    pts.push([x + randn() * 0.04, y + randn() * 0.04]);
  }
  return pts;
}

export function sampleChecker(n = 512) {
  const pts = [];
  const cells = [-1.2, -0.4, 0.4, 1.2];
  while (pts.length < n) {
    const ix = (Math.random() * cells.length) | 0;
    const iy = (Math.random() * cells.length) | 0;
    if ((ix + iy) % 2 === 0) {
      pts.push([
        cells[ix] + (Math.random() - 0.5) * 0.55,
        cells[iy] + (Math.random() - 0.5) * 0.55,
      ]);
    }
  }
  return pts;
}

export function sampleGaussians(n = 512) {
  const centers = [
    [-1.1, -1.0],
    [1.1, -1.0],
    [0, 1.15],
    [-1.2, 0.7],
    [1.2, 0.7],
    [0, -0.15],
  ];
  const pts = [];
  for (let i = 0; i < n; i++) {
    const c = centers[i % centers.length];
    pts.push([c[0] + randn() * 0.18, c[1] + randn() * 0.18]);
  }
  return pts;
}

export function sampleGaussian(n = 512, scale = 1.0) {
  const pts = [];
  for (let i = 0; i < n; i++) pts.push([randn() * scale, randn() * scale]);
  return pts;
}

const GENERATORS = {
  rings: sampleRings,
  moons: sampleMoons,
  spiral: sampleSpiral,
  swiss: sampleSwissRoll,
  checker: sampleChecker,
  gaussians: sampleGaussians,
  noise: (n) => sampleGaussian(n, 1.0),
};

export function getDataset(name, n = 512) {
  const fn = GENERATORS[name] || sampleRings;
  return fn(n);
}

export { randn, addNoise, GENERATORS };
