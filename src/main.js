/**
 * DiffLab — main application bootstrap & training loop
 */

import { getDataset, randn } from './data/datasets.js';
import { buildSchedule } from './models/schedules.js';
import { createTimeMLP } from './models/network.js';
import { ddpmTrainStep, sampleDDPM, fieldFromModel } from './models/ddpm.js';
import { fmTrainStep, sampleFM, fmField } from './models/flow_matching.js';
import { CanvasViz } from './viz/canvas_viz.js';
import { LossChart } from './viz/charts.js';

const T_DIM = 16;
const N_DATA = 512;
const N_GEN = 400;

const state = {
  tf: null,
  model: null,
  optimizer: null,
  schedule: null,
  dataFlat: null,
  genFlat: null,
  field: null,
  trajs: null,
  training: false,
  step: 0,
  loss: 0,
  objective: 'fm', // 'fm' | 'eps' | 'x0' | 'score'
  dataset: 'rings',
  T: 100,
  scheduleType: 'cosine',
  lr: 1e-3,
  hidden: 128,
  depth: 3,
  batch: 128,
  sampleSteps: 40,
  fieldT: 0.5,
  animating: false,
};

const els = {};

function $(id) { return document.getElementById(id); }

function flatFromPoints(pts) {
  const f = new Float32Array(pts.length * 2);
  for (let i = 0; i < pts.length; i++) {
    f[i * 2] = pts[i][0];
    f[i * 2 + 1] = pts[i][1];
  }
  return Array.from(f);
}

function sampleBatch(flat, batch) {
  const n = flat.length / 2;
  const out = new Float32Array(batch * 2);
  for (let i = 0; i < batch; i++) {
    const j = (Math.random() * n) | 0;
    out[i * 2] = flat[j * 2];
    out[i * 2 + 1] = flat[j * 2 + 1];
  }
  return Array.from(out);
}

async function loadTF() {
  // TensorFlow.js loaded via script tag as global
  if (!window.tf) throw new Error('TensorFlow.js not loaded');
  await window.tf.ready();
  // Prefer WebGL
  try {
    await window.tf.setBackend('webgl');
  } catch (_) {
    await window.tf.setBackend('cpu');
  }
  state.tf = window.tf;
  $('backend').textContent = state.tf.getBackend();
}

function rebuildModel() {
  const tf = state.tf;
  if (state.model) {
    state.model.dispose();
  }
  state.model = createTimeMLP({
    tf,
    hidden: state.hidden,
    depth: state.depth,
    tDim: T_DIM,
  });
  state.optimizer = tf.train.adam(state.lr);
  state.step = 0;
  state.lossChart?.reset();
  updateMetrics();
}

function loadDataset(name) {
  state.dataset = name;
  if (name === 'draw') {
    state.viz.drawMode = true;
    $('drawHint').classList.add('show');
    const drawn = state.viz.getDrawnAsDataset(N_DATA);
    if (drawn) {
      state.dataFlat = drawn;
    } else {
      // placeholder empty until user draws
      state.dataFlat = flatFromPoints(getDataset('noise', 32));
    }
  } else {
    state.viz.drawMode = false;
    $('drawHint').classList.remove('show');
    state.dataFlat = flatFromPoints(getDataset(name, N_DATA));
  }
  state.genFlat = null;
  state.field = null;
  state.trajs = null;
  render();
}

function rebuildSchedule() {
  state.schedule = buildSchedule(state.T, state.scheduleType);
}

async function trainOneStep() {
  const tf = state.tf;
  const batch = sampleBatch(state.dataFlat, state.batch);
  let loss;
  if (state.objective === 'fm') {
    loss = await fmTrainStep(tf, state.model, state.optimizer, batch, T_DIM);
  } else {
    loss = await ddpmTrainStep(
      tf, state.model, state.optimizer, batch, state.schedule, state.objective, T_DIM
    );
  }
  state.step += 1;
  state.loss = loss;
  state.lossChart.push(loss);
  return loss;
}

async function trainLoop() {
  while (state.training) {
    const start = performance.now();
    // several steps per frame for snappy feel
    const stepsPerFrame = state.objective === 'fm' ? 4 : 3;
    for (let i = 0; i < stepsPerFrame; i++) {
      await trainOneStep();
    }
    if (state.step % 20 < stepsPerFrame) {
      await refreshGenLight();
    }
    updateMetrics();
    // yield to UI
    const elapsed = performance.now() - start;
    await new Promise((r) => setTimeout(r, Math.max(0, 16 - elapsed)));
  }
}

async function refreshGenLight() {
  try {
    if (state.objective === 'fm') {
      state.genFlat = await sampleFM(state.tf, state.model, Math.min(N_GEN, 200), T_DIM, 25);
    } else {
      state.genFlat = await sampleDDPM(
        state.tf, state.model, Math.min(N_GEN, 150), state.schedule,
        state.objective, T_DIM, Math.min(30, state.T)
      );
    }
    if (state.viz.showField) await refreshField();
    render();
  } catch (e) {
    console.warn('sample refresh', e);
  }
}

async function fullSample() {
  setStatus('Sampling…', true);
  try {
    if (state.objective === 'fm') {
      state.genFlat = await sampleFM(
        state.tf, state.model, N_GEN, T_DIM, state.sampleSteps
      );
    } else {
      state.genFlat = await sampleDDPM(
        state.tf, state.model, N_GEN, state.schedule,
        state.objective, T_DIM, state.sampleSteps
      );
    }
    render();
    setStatus('Ready');
  } catch (e) {
    console.error(e);
    setStatus('Sample error');
  }
}

async function refreshField() {
  const tf = state.tf;
  const grid = 12;
  const xs = [], ys = [];
  for (let i = 0; i < grid; i++) {
    for (let j = 0; j < grid; j++) {
      const x = -1.9 + (3.8 * i) / (grid - 1);
      const y = -1.9 + (3.8 * j) / (grid - 1);
      xs.push(x);
      ys.push(y);
    }
  }
  let vTensor;
  if (state.objective === 'fm') {
    vTensor = fmField(tf, state.model, xs, ys, state.fieldT, T_DIM);
  } else {
    vTensor = fieldFromModel(
      tf, state.model, xs, ys, state.fieldT, state.objective, state.schedule, T_DIM
    );
  }
  const v = await vTensor.data();
  vTensor.dispose();
  state.field = xs.map((x, i) => ({
    x,
    y: ys[i],
    vx: v[i * 2],
    vy: v[i * 2 + 1],
  }));
}

async function computeTrajectories() {
  const tf = state.tf;
  const nTraj = 24;
  const steps = 30;
  const trajs = [];

  // start from noise
  let particles = [];
  for (let i = 0; i < nTraj; i++) {
    particles.push([randn(), randn()]);
  }

  for (let s = 0; s <= steps; s++) {
    trajs.push(particles.map((p) => [p[0], p[1]]));
    if (s === steps) break;

    if (state.objective === 'fm') {
      const tVal = s / steps;
      const flat = particles.flat();
      const xy = tf.tensor2d(flat, [nTraj, 2]);
      const t = tf.fill([nTraj]).add(tVal);
      const { predict } = await import('./models/network.js');
      const v = predict(tf, state.model, xy, t, T_DIM);
      const vd = await v.data();
      v.dispose();
      xy.dispose();
      const dt = 1 / steps;
      particles = particles.map((p, i) => [
        p[0] + vd[i * 2] * dt,
        p[1] + vd[i * 2 + 1] * dt,
      ]);
    } else {
      // coarse DDPM-style step using field approximation
      const tFrac = 1 - s / steps;
      const flat = particles.flat();
      const xy = tf.tensor2d(flat, [nTraj, 2]);
      const t = tf.fill([nTraj]).add(Math.max(tFrac, 0.02));
      const { predict } = await import('./models/network.js');
      const pred = predict(tf, state.model, xy, t, T_DIM);
      const pd = await pred.data();
      pred.dispose();
      xy.dispose();
      const dt = 0.08;
      particles = particles.map((p, i) => {
        let vx = -pd[i * 2];
        let vy = -pd[i * 2 + 1];
        if (state.objective === 'x0') {
          vx = pd[i * 2] - p[0];
          vy = pd[i * 2 + 1] - p[1];
        } else if (state.objective === 'score') {
          vx = pd[i * 2];
          vy = pd[i * 2 + 1];
        }
        return [p[0] + vx * dt, p[1] + vy * dt];
      });
    }
  }

  // reformat: trajs[step][particle] → trajs[particle][step]
  const byParticle = [];
  for (let p = 0; p < nTraj; p++) {
    byParticle.push(trajs.map((frame) => frame[p]));
  }
  state.trajs = byParticle;
  render();
}

async function animateForward() {
  if (state.animating) return;
  state.animating = true;
  setStatus('Forward noising…', true);
  const data = state.dataFlat.slice();
  const n = data.length / 2;
  const frames = 40;
  for (let f = 0; f <= frames; f++) {
    const t = f / frames;
    // x_t = sqrt(ā) x0 + sqrt(1-ā) ε  with ā ≈ (1-t)^2-ish for viz
    const ab = Math.max(1e-4, (1 - t) ** 2);
    const sa = Math.sqrt(ab);
    const so = Math.sqrt(1 - ab);
    const noisy = new Float32Array(data.length);
    for (let i = 0; i < n; i++) {
      noisy[i * 2] = sa * data[i * 2] + so * randn();
      noisy[i * 2 + 1] = sa * data[i * 2 + 1] + so * randn();
    }
    state.genFlat = Array.from(noisy);
    render();
    await new Promise((r) => setTimeout(r, 40));
  }
  state.animating = false;
  setStatus('Ready');
}

async function animateReverse() {
  if (state.animating) return;
  state.animating = true;
  setStatus('Generating…', true);
  const steps = state.sampleSteps;
  if (state.objective === 'fm') {
    let x = [];
    for (let i = 0; i < N_GEN; i++) x.push([randn(), randn()]);
    const tf = state.tf;
    const { predict } = await import('./models/network.js');
    for (let s = 0; s <= steps; s++) {
      state.genFlat = x.flat();
      render();
      if (s === steps) break;
      const tVal = s / steps;
      const xy = tf.tensor2d(x.flat(), [N_GEN, 2]);
      const t = tf.fill([N_GEN]).add(tVal);
      const v = predict(tf, state.model, xy, t, T_DIM);
      const vd = await v.data();
      v.dispose();
      xy.dispose();
      const dt = 1 / steps;
      x = x.map((p, i) => [p[0] + vd[i * 2] * dt, p[1] + vd[i * 2 + 1] * dt]);
      await new Promise((r) => setTimeout(r, 25));
    }
  } else {
    // stepwise DDPM sample with intermediate renders
    const tf = state.tf;
    let xt = tf.randomNormal([Math.min(N_GEN, 250), 2]);
    const T = state.schedule.T;
    const stride = Math.max(1, Math.floor(T / steps));
    const { ddpmPSampleStep } = await import('./models/ddpm.js');
    for (let t = T - 1; t >= 0; t -= stride) {
      const next = ddpmPSampleStep(
        tf, state.model, xt, t, state.schedule, state.objective, T_DIM
      );
      xt.dispose();
      xt = next;
      if (t % (stride * 2) === 0 || t < stride) {
        const d = await xt.data();
        state.genFlat = Array.from(d);
        render();
        await new Promise((r) => setTimeout(r, 20));
      }
    }
    const d = await xt.data();
    state.genFlat = Array.from(d);
    xt.dispose();
    render();
  }
  state.animating = false;
  setStatus('Ready');
}

function render() {
  state.viz.render({
    data: state.dataFlat,
    gen: state.genFlat,
    field: state.field,
    trajs: state.trajs,
  });
}

function updateMetrics() {
  $('mStep').textContent = String(state.step);
  $('mLoss').textContent = state.loss ? state.loss.toFixed(4) : '—';
  $('mObj').textContent = labelObjective(state.objective);
  $('mBackend').textContent = state.tf ? state.tf.getBackend() : '—';
  $('statusText').innerHTML = state.training
    ? `<span class="spinner"></span>Training · step <strong>${state.step}</strong> · loss <strong>${state.loss.toFixed(4)}</strong>`
    : `Ready · step <strong>${state.step}</strong>`;
}

function labelObjective(o) {
  return ({
    fm: 'Flow Matching (v)',
    eps: 'DDPM (ε-pred)',
    x0: 'x₀ prediction',
    score: 'Score matching',
  })[o] || o;
}

function setStatus(msg, busy = false) {
  $('statusText').innerHTML = busy
    ? `<span class="spinner"></span>${msg}`
    : msg;
}

function setChipGroup(groupEl, value) {
  groupEl.querySelectorAll('.chip').forEach((c) => {
    c.classList.toggle('active', c.dataset.value === value);
  });
}

function wireUI() {
  // datasets
  $('datasetChips').addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    setChipGroup($('datasetChips'), chip.dataset.value);
    if (chip.dataset.value === 'draw') {
      state.viz.clearDrawing();
    }
    loadDataset(chip.dataset.value);
  });

  // objectives
  $('objectiveChips').addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    setChipGroup($('objectiveChips'), chip.dataset.value);
    state.objective = chip.dataset.value;
    $('eqBox').textContent = equations(state.objective);
    updateMetrics();
  });

  // schedule
  $('scheduleSelect').addEventListener('change', (e) => {
    state.scheduleType = e.target.value;
    rebuildSchedule();
  });

  // sliders
  const bindRange = (id, key, fmt, onChange) => {
    const el = $(id);
    const val = $(id + 'Val');
    el.addEventListener('input', () => {
      state[key] = fmt(el.value);
      val.textContent = String(state[key]);
      onChange?.();
    });
  };
  bindRange('lrRange', 'lr', (v) => Number(v), () => {
    if (state.optimizer) state.optimizer.learningRate = state.lr;
  });
  // lr uses log scale via select-like values
  $('lrRange').addEventListener('input', () => {
    const exp = Number($('lrRange').value);
    state.lr = Math.pow(10, exp);
    $('lrRangeVal').textContent = state.lr.toExponential(0);
    if (state.optimizer) {
      // recreate optimizer with new lr
      state.optimizer = state.tf.train.adam(state.lr);
    }
  });
  bindRange('tRange', 'T', (v) => Number(v) | 0, rebuildSchedule);
  bindRange('hiddenRange', 'hidden', (v) => Number(v) | 0);
  bindRange('depthRange', 'depth', (v) => Number(v) | 0);
  bindRange('batchRange', 'batch', (v) => Number(v) | 0);
  bindRange('stepsRange', 'sampleSteps', (v) => Number(v) | 0);
  bindRange('fieldTRange', 'fieldT', (v) => Number(v), async () => {
    if (state.viz.showField) {
      await refreshField();
      render();
    }
  });

  $('btnTrain').addEventListener('click', async () => {
    if (state.training) {
      state.training = false;
      $('btnTrain').textContent = 'Train';
      $('btnTrain').classList.add('primary');
      updateMetrics();
      return;
    }
    // if draw mode, pull points
    if (state.dataset === 'draw') {
      const d = state.viz.getDrawnAsDataset(N_DATA);
      if (!d || d.length < 32) {
        alert('Draw more points on the canvas first (click & drag).');
        return;
      }
      state.dataFlat = d;
    }
    state.training = true;
    $('btnTrain').textContent = 'Stop';
    $('btnTrain').classList.remove('primary');
    trainLoop();
  });

  $('btnReset').addEventListener('click', () => {
    state.training = false;
    $('btnTrain').textContent = 'Train';
    $('btnTrain').classList.add('primary');
    rebuildModel();
    state.genFlat = null;
    state.field = null;
    state.trajs = null;
    render();
    updateMetrics();
  });

  $('btnSample').addEventListener('click', () => fullSample());
  $('btnAnimRev').addEventListener('click', () => animateReverse());
  $('btnAnimFwd').addEventListener('click', () => animateForward());
  $('btnTraj').addEventListener('click', async () => {
    setStatus('Computing trajectories…', true);
    await computeTrajectories();
    state.viz.showTraj = true;
    $('togTraj').checked = true;
    setStatus('Ready');
  });

  $('btnApplyArch').addEventListener('click', () => {
    state.training = false;
    $('btnTrain').textContent = 'Train';
    rebuildModel();
    setStatus('Model rebuilt');
  });

  $('togData').addEventListener('change', (e) => {
    state.viz.showData = e.target.checked;
    render();
  });
  $('togGen').addEventListener('change', (e) => {
    state.viz.showGen = e.target.checked;
    render();
  });
  $('togField').addEventListener('change', async (e) => {
    state.viz.showField = e.target.checked;
    if (e.target.checked) await refreshField();
    render();
  });
  $('togTraj').addEventListener('change', (e) => {
    state.viz.showTraj = e.target.checked;
    render();
  });

  $('btnClearDraw').addEventListener('click', () => {
    state.viz.clearDrawing();
    render();
  });

  window.addEventListener('resize', () => {
    state.viz.resize();
    render();
  });
}

function equations(obj) {
  if (obj === 'fm') {
    return `Flow Matching (linear OT path)
x_t = (1−t)·ε + t·x₁
v*  = x₁ − ε
L   = E ‖ v_θ(x_t,t) − v* ‖²

Sample: ODE  dx/dt = v_θ(x,t),  t: 0→1`;
  }
  if (obj === 'eps') {
    return `DDPM ε-prediction
q(x_t|x₀) = N(√ā_t x₀, (1−ā_t)I)
L = E ‖ ε − ε_θ(x_t,t) ‖²

Reverse: ancestral / DDIM using ε_θ`;
  }
  if (obj === 'x0') {
    return `x₀ prediction
Network predicts clean sample x₀
L = E ‖ x₀ − x̂₀(x_t,t) ‖²

Useful reparameterization of DDPM`;
  }
  return `Denoising score matching
score s ≈ ∇log p_t(x) ≈ −ε / σ_t
L = E ‖ s_θ(x_t,t) − (−ε/σ_t) ‖²

Score SDE / probability-flow ODE`;
}

async function main() {
  els.canvas = $('mainCanvas');
  state.viz = new CanvasViz(els.canvas);
  state.lossChart = new LossChart($('lossChart'));
  // fix chart resolution
  const lc = $('lossChart');
  lc.width = lc.clientWidth * 2 || 560;
  lc.height = 120 * 2;
  state.lossChart.draw();

  setStatus('Loading TensorFlow.js…', true);
  await loadTF();
  rebuildSchedule();
  rebuildModel();
  loadDataset('rings');
  wireUI();
  $('eqBox').textContent = equations('fm');
  setStatus('Ready — hit Train');
  updateMetrics();
  render();

  // initial random gen cloud
  const noise = [];
  for (let i = 0; i < N_GEN; i++) noise.push(randn(), randn());
  state.genFlat = noise;
  render();
}

main().catch((e) => {
  console.error(e);
  setStatus('Failed to start: ' + e.message);
});
