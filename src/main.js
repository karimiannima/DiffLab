/**
 * DiffLab — main application bootstrap & training loop
 * Covers: DDPM, Flow Matching, Score-based SDEs, Latent Diffusion (LDM)
 */

import { getDataset, randn } from './data/datasets.js';
import { buildSchedule } from './models/schedules.js';
import { createTimeMLP } from './models/network.js';
import { ddpmTrainStep, sampleDDPM, fieldFromModel, ddpmPSampleStep } from './models/ddpm.js';
import { fmTrainStep, sampleFM, fmField } from './models/flow_matching.js';
import { scoreTrainStep, sampleScoreSDE, scoreField } from './models/score_sde.js';
import {
  createAutoencoder,
  aeTrainStep,
  encodeFlat,
  createLatentDiffModel,
  ldmDiffTrainStep,
  sampleLDM,
} from './models/ldm.js';
import { CanvasViz } from './viz/canvas_viz.js';
import { LossChart } from './viz/charts.js';

const T_DIM = 16;
const N_DATA = 512;
const N_GEN = 400;
const LATENT_DIM = 2;

const state = {
  tf: null,
  model: null,
  optimizer: null,
  // LDM components
  encoder: null,
  decoder: null,
  aeOptimizer: null,
  latentModel: null,
  latentOpt: null,
  latentData: null, // cached encoded dataset
  schedule: null,
  dataFlat: null,
  genFlat: null,
  latentGenFlat: null,
  field: null,
  trajs: null,
  training: false,
  step: 0,
  loss: 0,
  aeLoss: 0,
  objective: 'fm',
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
  showLatent: false,
  aeWarmup: 0,
};

function $(id) { return document.getElementById(id); }

function isScoreSde(obj) {
  return obj === 'score_ve' || obj === 'score_vp' || obj === 'score';
}
function isLdm(obj) {
  return obj === 'ldm_eps' || obj === 'ldm_fm';
}

function flatFromPoints(pts) {
  const f = new Float32Array(pts.length * 2);
  for (let i = 0; i < pts.length; i++) {
    f[i * 2] = pts[i][0];
    f[i * 2 + 1] = pts[i][1];
  }
  return Array.from(f);
}

function sampleBatch(flat, batch, dim = 2) {
  const n = flat.length / dim;
  const out = new Float32Array(batch * dim);
  for (let i = 0; i < batch; i++) {
    const j = (Math.random() * n) | 0;
    for (let d = 0; d < dim; d++) out[i * dim + d] = flat[j * dim + d];
  }
  return Array.from(out);
}

async function loadTF() {
  if (!window.tf) throw new Error('TensorFlow.js not loaded');
  await window.tf.ready();
  try {
    await window.tf.setBackend('webgl');
  } catch (_) {
    await window.tf.setBackend('cpu');
  }
  state.tf = window.tf;
  $('backend').textContent = state.tf.getBackend();
}

function disposeModel(m) {
  if (m) {
    try { m.dispose(); } catch (_) { /* */ }
  }
}

function rebuildModel() {
  const tf = state.tf;
  disposeModel(state.model);
  disposeModel(state.encoder);
  disposeModel(state.decoder);
  disposeModel(state.latentModel);

  state.model = createTimeMLP({
    tf,
    hidden: state.hidden,
    depth: state.depth,
    tDim: T_DIM,
    outDim: 2,
  });
  state.optimizer = tf.train.adam(state.lr);

  // LDM stack
  const ae = createAutoencoder(tf, { latentDim: LATENT_DIM, hidden: Math.max(32, state.hidden / 2) });
  state.encoder = ae.encoder;
  state.decoder = ae.decoder;
  state.aeOptimizer = tf.train.adam(state.lr);
  state.latentModel = createLatentDiffModel(tf, {
    latentDim: LATENT_DIM,
    hidden: state.hidden,
    depth: state.depth,
    tDim: T_DIM,
  });
  state.latentOpt = tf.train.adam(state.lr);
  state.latentData = null;
  state.aeWarmup = 0;

  state.step = 0;
  state.loss = 0;
  state.aeLoss = 0;
  state.lossChart?.reset();
  updateMetrics();
  updateLdmPanel();
}

function rebuildSchedule() {
  state.schedule = buildSchedule(state.T, state.scheduleType);
}

function loadDataset(name) {
  state.dataset = name;
  if (name === 'draw') {
    state.viz.drawMode = true;
    $('drawHint').classList.add('show');
    const drawn = state.viz.getDrawnAsDataset(N_DATA);
    state.dataFlat = drawn || flatFromPoints(getDataset('noise', 32));
  } else {
    state.viz.drawMode = false;
    $('drawHint').classList.remove('show');
    state.dataFlat = flatFromPoints(getDataset(name, N_DATA));
  }
  state.genFlat = null;
  state.latentGenFlat = null;
  state.latentData = null;
  state.field = null;
  state.trajs = null;
  render();
}

async function ensureLatentCache() {
  if (!state.latentData && state.encoder) {
    state.latentData = await encodeFlat(state.tf, state.encoder, state.dataFlat);
  }
}

async function trainOneStep() {
  const tf = state.tf;
  const obj = state.objective;
  let loss;

  if (isLdm(obj)) {
    // Warm up AE then joint train
    const xBatch = sampleBatch(state.dataFlat, state.batch, 2);
    if (state.aeWarmup < 80 || state.step % 3 === 0) {
      state.aeLoss = await aeTrainStep(tf, state.encoder, state.decoder, state.aeOptimizer, xBatch);
      state.aeWarmup += 1;
      state.latentData = null; // invalidate
    }
    await ensureLatentCache();
    // re-encode batch for diffusion
    const zBatch = await encodeFlat(tf, state.encoder, xBatch);
    const mode = obj === 'ldm_fm' ? 'fm' : 'eps';
    loss = await ldmDiffTrainStep(
      tf, state.latentModel, state.latentOpt, zBatch, LATENT_DIM,
      state.schedule, mode, T_DIM
    );
  } else if (obj === 'score_ve') {
    const batch = sampleBatch(state.dataFlat, state.batch);
    loss = await scoreTrainStep(tf, state.model, state.optimizer, batch, T_DIM, 've');
  } else if (obj === 'score_vp') {
    const batch = sampleBatch(state.dataFlat, state.batch);
    loss = await scoreTrainStep(tf, state.model, state.optimizer, batch, T_DIM, 'vp');
  } else if (obj === 'score') {
    const batch = sampleBatch(state.dataFlat, state.batch);
    loss = await ddpmTrainStep(
      tf, state.model, state.optimizer, batch, state.schedule, 'score', T_DIM
    );
  } else if (obj === 'fm') {
    const batch = sampleBatch(state.dataFlat, state.batch);
    loss = await fmTrainStep(tf, state.model, state.optimizer, batch, T_DIM);
  } else {
    const batch = sampleBatch(state.dataFlat, state.batch);
    loss = await ddpmTrainStep(
      tf, state.model, state.optimizer, batch, state.schedule, obj, T_DIM
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
    const stepsPerFrame = isLdm(state.objective) ? 2 : (state.objective === 'fm' ? 4 : 3);
    for (let i = 0; i < stepsPerFrame; i++) {
      await trainOneStep();
    }
    if (state.step % 24 < stepsPerFrame) {
      await refreshGenLight();
    }
    updateMetrics();
    const elapsed = performance.now() - start;
    await new Promise((r) => setTimeout(r, Math.max(0, 16 - elapsed)));
  }
}

async function refreshGenLight() {
  try {
    const n = Math.min(N_GEN, 180);
    if (isLdm(state.objective)) {
      const mode = state.objective === 'ldm_fm' ? 'fm' : 'eps';
      const { xFlat, zFlat } = await sampleLDM(
        state.tf, state.latentModel, state.decoder, n, LATENT_DIM,
        state.schedule, mode, T_DIM, 20
      );
      state.genFlat = xFlat;
      state.latentGenFlat = zFlat;
    } else if (state.objective === 'score_ve') {
      state.genFlat = await sampleScoreSDE(state.tf, state.model, n, T_DIM, 've', 25);
    } else if (state.objective === 'score_vp') {
      state.genFlat = await sampleScoreSDE(state.tf, state.model, n, T_DIM, 'vp', 25);
    } else if (state.objective === 'fm') {
      state.genFlat = await sampleFM(state.tf, state.model, n, T_DIM, 25);
    } else {
      state.genFlat = await sampleDDPM(
        state.tf, state.model, Math.min(n, 150), state.schedule,
        state.objective === 'score' ? 'score' : state.objective, T_DIM, Math.min(30, state.T)
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
    if (isLdm(state.objective)) {
      const mode = state.objective === 'ldm_fm' ? 'fm' : 'eps';
      const { xFlat, zFlat } = await sampleLDM(
        state.tf, state.latentModel, state.decoder, N_GEN, LATENT_DIM,
        state.schedule, mode, T_DIM, state.sampleSteps
      );
      state.genFlat = xFlat;
      state.latentGenFlat = zFlat;
    } else if (state.objective === 'score_ve') {
      state.genFlat = await sampleScoreSDE(
        state.tf, state.model, N_GEN, T_DIM, 've', state.sampleSteps
      );
    } else if (state.objective === 'score_vp') {
      state.genFlat = await sampleScoreSDE(
        state.tf, state.model, N_GEN, T_DIM, 'vp', state.sampleSteps
      );
    } else if (state.objective === 'fm') {
      state.genFlat = await sampleFM(
        state.tf, state.model, N_GEN, T_DIM, state.sampleSteps
      );
    } else {
      state.genFlat = await sampleDDPM(
        state.tf, state.model, N_GEN, state.schedule,
        state.objective === 'score' ? 'score' : state.objective, T_DIM, state.sampleSteps
      );
    }
    render();
    setStatus('Ready');
  } catch (e) {
    console.error(e);
    setStatus('Sample error: ' + e.message);
  }
}

async function refreshField() {
  const tf = state.tf;
  const grid = 12;
  const xs = [], ys = [];
  for (let i = 0; i < grid; i++) {
    for (let j = 0; j < grid; j++) {
      xs.push(-1.9 + (3.8 * i) / (grid - 1));
      ys.push(-1.9 + (3.8 * j) / (grid - 1));
    }
  }

  let vTensor;
  if (isLdm(state.objective)) {
    // field in data space via score/velocity of decoded path is hard;
    // show field from main model if available, else skip
    // Use latent model mapped: approximate by encoding grid → latent field → skip
    // For viz: run data-space model field as zero / use decoder Jacobian-free: encode, get latent v, decode finite-diff
    try {
      const flat = [];
      for (let i = 0; i < xs.length; i++) flat.push(xs[i], ys[i]);
      const zFlat = await encodeFlat(tf, state.encoder, flat);
      // latent velocity / -eps
      const { predictLatent } = await import('./models/ldm.js');
      const B = xs.length;
      const z = tf.tensor2d(zFlat, [B, LATENT_DIM]);
      const t = tf.fill([B]).add(state.fieldT);
      const pred = predictLatent(tf, state.latentModel, z, t, T_DIM, LATENT_DIM);
      const pd = await pred.data();
      pred.dispose();
      z.dispose();
      // map latent vectors as display field (z-space arrows at data positions — educational)
      state.field = xs.map((x, i) => ({
        x, y: ys[i],
        vx: pd[i * LATENT_DIM],
        vy: pd[i * LATENT_DIM + 1] || 0,
      }));
      return;
    } catch (e) {
      console.warn(e);
      state.field = [];
      return;
    }
  } else if (state.objective === 'fm') {
    vTensor = fmField(tf, state.model, xs, ys, state.fieldT, T_DIM);
  } else if (isScoreSde(state.objective)) {
    vTensor = scoreField(tf, state.model, xs, ys, state.fieldT, T_DIM);
  } else {
    vTensor = fieldFromModel(
      tf, state.model, xs, ys, state.fieldT,
      state.objective === 'score' ? 'score' : state.objective,
      state.schedule, T_DIM
    );
  }
  const v = await vTensor.data();
  vTensor.dispose();
  state.field = xs.map((x, i) => ({
    x, y: ys[i], vx: v[i * 2], vy: v[i * 2 + 1],
  }));
}

async function computeTrajectories() {
  const tf = state.tf;
  const nTraj = 24;
  const steps = 30;
  const trajs = [];
  let particles = [];
  for (let i = 0; i < nTraj; i++) particles.push([randn() * (isScoreSde(state.objective) ? 1.5 : 1), randn() * (isScoreSde(state.objective) ? 1.5 : 1)]);

  const { predict } = await import('./models/network.js');

  for (let s = 0; s <= steps; s++) {
    trajs.push(particles.map((p) => [p[0], p[1]]));
    if (s === steps) break;

    if (isLdm(state.objective)) {
      // encode → latent step → decode for visualization
      const flat = particles.flat();
      const zFlat = await encodeFlat(tf, state.encoder, flat);
      const { predictLatent } = await import('./models/ldm.js');
      const z = tf.tensor2d(zFlat, [nTraj, LATENT_DIM]);
      const tVal = s / steps;
      const t = tf.fill([nTraj]).add(state.objective === 'ldm_fm' ? tVal : 1 - tVal);
      const pred = predictLatent(tf, state.latentModel, z, t, T_DIM, LATENT_DIM);
      const pd = await pred.data();
      pred.dispose();
      z.dispose();
      const dt = 1 / steps;
      let zNew = [];
      for (let i = 0; i < nTraj; i++) {
        if (state.objective === 'ldm_fm') {
          zNew.push([
            zFlat[i * 2] + pd[i * 2] * dt,
            zFlat[i * 2 + 1] + pd[i * 2 + 1] * dt,
          ]);
        } else {
          zNew.push([
            zFlat[i * 2] - pd[i * 2] * 0.1,
            zFlat[i * 2 + 1] - pd[i * 2 + 1] * 0.1,
          ]);
        }
      }
      const { decodeFlat } = await import('./models/ldm.js');
      const xFlat = await decodeFlat(tf, state.decoder, zNew.flat(), LATENT_DIM);
      particles = [];
      for (let i = 0; i < nTraj; i++) particles.push([xFlat[i * 2], xFlat[i * 2 + 1]]);
    } else if (state.objective === 'fm') {
      const tVal = s / steps;
      const xy = tf.tensor2d(particles.flat(), [nTraj, 2]);
      const t = tf.fill([nTraj]).add(tVal);
      const v = predict(tf, state.model, xy, t, T_DIM);
      const vd = await v.data();
      v.dispose(); xy.dispose();
      const dt = 1 / steps;
      particles = particles.map((p, i) => [p[0] + vd[i * 2] * dt, p[1] + vd[i * 2 + 1] * dt]);
    } else {
      // score / DDPM-style field integration
      const tFrac = Math.max(0.05, 1 - s / steps);
      const xy = tf.tensor2d(particles.flat(), [nTraj, 2]);
      const t = tf.fill([nTraj]).add(tFrac);
      const pred = predict(tf, state.model, xy, t, T_DIM);
      const pd = await pred.data();
      pred.dispose(); xy.dispose();
      const dt = 0.1;
      particles = particles.map((p, i) => {
        let vx = pd[i * 2];
        let vy = pd[i * 2 + 1];
        if (state.objective === 'eps' || state.objective === 'x0') {
          if (state.objective === 'x0') {
            vx = pd[i * 2] - p[0];
            vy = pd[i * 2 + 1] - p[1];
          } else {
            vx = -pd[i * 2];
            vy = -pd[i * 2 + 1];
          }
        }
        // score points uphill density
        return [p[0] + vx * dt, p[1] + vy * dt];
      });
    }
  }

  state.trajs = [];
  for (let p = 0; p < nTraj; p++) {
    state.trajs.push(trajs.map((frame) => frame[p]));
  }
  state.viz.showTraj = true;
  $('togTraj').checked = true;
  render();
}

async function animateForward() {
  if (state.animating) return;
  state.animating = true;
  setStatus(isLdm(state.objective) ? 'LDM forward: encode → noise latent…' : 'Forward noising…', true);
  const data = state.dataFlat.slice();
  const n = data.length / 2;
  const frames = 40;

  if (isLdm(state.objective)) {
    await ensureLatentCache();
    const zData = state.latentData || await encodeFlat(state.tf, state.encoder, data);
    const zn = zData.length / LATENT_DIM;
    for (let f = 0; f <= frames; f++) {
      const t = f / frames;
      const ab = Math.max(1e-4, (1 - t) ** 2);
      const sa = Math.sqrt(ab);
      const so = Math.sqrt(1 - ab);
      const noisyZ = new Float32Array(zData.length);
      for (let i = 0; i < zn; i++) {
        for (let d = 0; d < LATENT_DIM; d++) {
          noisyZ[i * LATENT_DIM + d] = sa * zData[i * LATENT_DIM + d] + so * randn();
        }
      }
      const { decodeFlat } = await import('./models/ldm.js');
      state.genFlat = await decodeFlat(state.tf, state.decoder, Array.from(noisyZ), LATENT_DIM);
      state.latentGenFlat = Array.from(noisyZ);
      render();
      await new Promise((r) => setTimeout(r, 35));
    }
  } else {
    for (let f = 0; f <= frames; f++) {
      const t = f / frames;
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
  }
  state.animating = false;
  setStatus('Ready');
}

async function animateReverse() {
  if (state.animating) return;
  state.animating = true;
  setStatus(isLdm(state.objective) ? 'LDM sampling in latent → decode…' : 'Generating…', true);
  const steps = state.sampleSteps;
  const tf = state.tf;

  if (isLdm(state.objective)) {
    const mode = state.objective === 'ldm_fm' ? 'fm' : 'eps';
    // animate intermediate by partial steps
    let z = tf.randomNormal([Math.min(N_GEN, 250), LATENT_DIM]);
    if (mode === 'fm') {
      const { predictLatent } = await import('./models/ldm.js');
      for (let s = 0; s <= steps; s++) {
        const x = state.decoder.predict(z);
        const xd = await x.data();
        x.dispose();
        state.genFlat = Array.from(xd);
        render();
        if (s === steps) break;
        const tVal = s / steps;
        const next = tf.tidy(() => {
          const t = tf.fill([z.shape[0]]).add(tVal);
          const v = predictLatent(tf, state.latentModel, z, t, T_DIM, LATENT_DIM);
          return z.add(v.mul(1 / steps));
        });
        z.dispose();
        z = next;
        await new Promise((r) => setTimeout(r, 25));
      }
    } else {
      const T = state.schedule.T;
      const stride = Math.max(1, Math.floor(T / steps));
      const { predictLatent } = await import('./models/ldm.js');
      for (let t = T - 1; t >= 0; t -= stride) {
        const next = tf.tidy(() => {
          const beta = state.schedule.betas[t];
          const alpha = state.schedule.alphas[t];
          const alphabar = state.schedule.alphasCumprod[t];
          const alphabarPrev = t > 0 ? state.schedule.alphasCumprod[t - 1] : 1;
          const tCont = tf.fill([z.shape[0]]).add((t + 1) / T);
          const eps = predictLatent(tf, state.latentModel, z, tCont, T_DIM, LATENT_DIM);
          const sqrtAb = Math.sqrt(alphabar);
          const sqrtOm = Math.sqrt(1 - alphabar);
          const x0pred = z.sub(eps.mul(sqrtOm)).div(sqrtAb + 1e-8);
          const coefX0 = (Math.sqrt(alphabarPrev) * beta) / (1 - alphabar + 1e-8);
          const coefXt = (Math.sqrt(alpha) * (1 - alphabarPrev)) / (1 - alphabar + 1e-8);
          const mean = x0pred.mul(coefX0).add(z.mul(coefXt));
          if (t === 0) return mean;
          return mean.add(tf.randomNormal(z.shape).mul(Math.sqrt(beta)));
        });
        z.dispose();
        z = next;
        if (t % (stride * 2) === 0 || t < stride) {
          const x = state.decoder.predict(z);
          const xd = await x.data();
          x.dispose();
          state.genFlat = Array.from(xd);
          render();
          await new Promise((r) => setTimeout(r, 20));
        }
      }
    }
    const zd = await z.data();
    state.latentGenFlat = Array.from(zd);
    z.dispose();
  } else if (state.objective === 'fm') {
    const { predict } = await import('./models/network.js');
    let x = [];
    for (let i = 0; i < N_GEN; i++) x.push([randn(), randn()]);
    for (let s = 0; s <= steps; s++) {
      state.genFlat = x.flat();
      render();
      if (s === steps) break;
      const tVal = s / steps;
      const xy = tf.tensor2d(x.flat(), [N_GEN, 2]);
      const t = tf.fill([N_GEN]).add(tVal);
      const v = predict(tf, state.model, xy, t, T_DIM);
      const vd = await v.data();
      v.dispose(); xy.dispose();
      const dt = 1 / steps;
      x = x.map((p, i) => [p[0] + vd[i * 2] * dt, p[1] + vd[i * 2 + 1] * dt]);
      await new Promise((r) => setTimeout(r, 25));
    }
  } else if (isScoreSde(state.objective)) {
    const sde = state.objective === 'score_vp' ? 'vp' : 've';
    // stepwise sample with renders
    for (let s = 5; s <= steps; s += Math.max(1, Math.floor(steps / 12))) {
      state.genFlat = await sampleScoreSDE(tf, state.model, Math.min(N_GEN, 200), T_DIM, sde, s);
      render();
      await new Promise((r) => setTimeout(r, 40));
    }
    state.genFlat = await sampleScoreSDE(tf, state.model, N_GEN, T_DIM, sde, steps);
    render();
  } else {
    let xt = tf.randomNormal([Math.min(N_GEN, 250), 2]);
    const T = state.schedule.T;
    const stride = Math.max(1, Math.floor(T / steps));
    const obj = state.objective === 'score' ? 'score' : state.objective;
    for (let t = T - 1; t >= 0; t -= stride) {
      const next = ddpmPSampleStep(tf, state.model, xt, t, state.schedule, obj, T_DIM);
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
  // optionally show latent as gen overlay when toggled
  let gen = state.genFlat;
  let data = state.dataFlat;
  if (state.showLatent && isLdm(state.objective)) {
    // show latent clouds instead
    data = state.latentData;
    gen = state.latentGenFlat;
  }
  state.viz.render({
    data,
    gen,
    field: state.field,
    trajs: state.trajs,
  });
}

function updateLdmPanel() {
  const panel = $('ldmInfo');
  if (!panel) return;
  if (isLdm(state.objective)) {
    panel.style.display = 'block';
    panel.innerHTML = `
      <strong>LDM pipeline</strong><br/>
      <code>x → E(x)=z → diffuse in z → D(z)=x</code><br/>
      Latent dim: ${LATENT_DIM} · AE loss: <span id="aeLossLive">${state.aeLoss ? state.aeLoss.toFixed(4) : '—'}</span><br/>
      <span class="muted">Toggle “Show latent space” to plot z instead of x.</span>
    `;
  } else if (isScoreSde(state.objective)) {
    panel.style.display = 'block';
    panel.innerHTML = `
      <strong>Score-based SDE</strong><br/>
      Network predicts <code>s_θ ≈ ∇log p_t(x)</code><br/>
      ${state.objective === 'score_ve' ? 'VE-SDE: x_t = x₀ + σ(t)ε' : state.objective === 'score_vp' ? 'VP-SDE: variance-preserving path' : 'Discrete DSM (DDPM-linked)'}
    `;
  } else {
    panel.style.display = 'none';
  }
}

function updateMetrics() {
  $('mStep').textContent = String(state.step);
  $('mLoss').textContent = state.loss ? state.loss.toFixed(4) : '—';
  $('mObj').textContent = labelObjective(state.objective);
  $('mBackend').textContent = state.tf ? state.tf.getBackend() : '—';
  const aeEl = $('aeLossLive');
  if (aeEl) aeEl.textContent = state.aeLoss ? state.aeLoss.toFixed(4) : '—';
  $('statusText').innerHTML = state.training
    ? `<span class="spinner"></span>Training · step <strong>${state.step}</strong> · loss <strong>${state.loss.toFixed(4)}</strong>`
    : `Ready · step <strong>${state.step}</strong>`;
}

function labelObjective(o) {
  return ({
    fm: 'Flow Matching (v)',
    eps: 'DDPM (ε-pred)',
    x0: 'x₀ prediction',
    score: 'Score (DSM)',
    score_ve: 'Score VE-SDE',
    score_vp: 'Score VP-SDE',
    ldm_eps: 'LDM (latent DDPM)',
    ldm_fm: 'LDM (latent FM)',
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

function equations(obj) {
  const map = {
    fm: `Flow Matching (linear OT path)
x_t = (1−t)·ε + t·x₁
v*  = x₁ − ε
L   = E ‖ v_θ(x_t,t) − v* ‖²

Sample: ODE dx/dt = v_θ , t: 0→1`,
    eps: `DDPM ε-prediction
q(x_t|x₀)=N(√ā_t x₀,(1−ā_t)I)
L = E ‖ ε − ε_θ(x_t,t) ‖²

Reverse: ancestral / DDIM`,
    x0: `x₀ prediction
L = E ‖ x₀ − x̂₀(x_t,t) ‖²
Reparameterization of DDPM`,
    score: `Denoising Score Matching
s ≈ ∇log p_t(x) ≈ −ε/σ_t
L = E ‖ s_θ − (−ε/σ) ‖²`,
    score_ve: `Score VE-SDE (Song et al.)
x_t = x₀ + σ(t) ε
σ(t)=σ_min (σ_max/σ_min)^t
target score = −ε/σ(t)
Sample: reverse SDE + Langevin`,
    score_vp: `Score VP-SDE
Variance-preserving diffusion
x_t = √ā(t) x₀ + √(1−ā) ε
s_θ ≈ −ε/√(1−ā)
Same family as DDPM continuous`,
    ldm_eps: `Latent Diffusion (LDM)
1) AE: x —E→ z —D→ x̂
2) DDPM on z: L = E‖ε−ε_θ(z_t,t)‖²
3) Sample z then x = D(z)

As in Stable Diffusion (Rombach et al.)`,
    ldm_fm: `Latent Flow Matching
1) AE: x —E→ z —D→ x̂
2) FM velocity in z-space
3) ODE sample z → decode x=D(z)

Modern latent generators (e.g. SD3/FLUX style)`,
  };
  return map[obj] || '';
}

function wireUI() {
  $('datasetChips').addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    setChipGroup($('datasetChips'), chip.dataset.value);
    if (chip.dataset.value === 'draw') state.viz.clearDrawing();
    loadDataset(chip.dataset.value);
  });

  $('objectiveChips').addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    setChipGroup($('objectiveChips'), chip.dataset.value);
    state.objective = chip.dataset.value;
    $('eqBox').textContent = equations(state.objective);
    updateMetrics();
    updateLdmPanel();
    // show latent toggle only for LDM
    const latRow = $('latentToggleRow');
    if (latRow) latRow.style.display = isLdm(state.objective) ? 'block' : 'none';
  });

  $('scheduleSelect').addEventListener('change', (e) => {
    state.scheduleType = e.target.value;
    rebuildSchedule();
  });

  $('lrRange').addEventListener('input', () => {
    const exp = Number($('lrRange').value);
    state.lr = Math.pow(10, exp);
    $('lrRangeVal').textContent = state.lr.toExponential(0);
    if (state.tf) {
      state.optimizer = state.tf.train.adam(state.lr);
      state.aeOptimizer = state.tf.train.adam(state.lr);
      state.latentOpt = state.tf.train.adam(state.lr);
    }
  });

  const bindRange = (id, key, fmt, onChange) => {
    const el = $(id);
    const val = $(id + 'Val');
    el.addEventListener('input', () => {
      state[key] = fmt(el.value);
      val.textContent = String(state[key]);
      onChange?.();
    });
  };
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
    if (state.dataset === 'draw') {
      const d = state.viz.getDrawnAsDataset(N_DATA);
      if (!d || d.length < 32) {
        alert('Draw more points on the canvas first (click & drag).');
        return;
      }
      state.dataFlat = d;
      state.latentData = null;
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
    state.latentGenFlat = null;
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
    setStatus('Ready');
  });

  $('btnApplyArch').addEventListener('click', () => {
    state.training = false;
    $('btnTrain').textContent = 'Train';
    rebuildModel();
    setStatus('Model rebuilt');
  });

  $('togData').addEventListener('change', (e) => {
    state.viz.showData = e.target.checked; render();
  });
  $('togGen').addEventListener('change', (e) => {
    state.viz.showGen = e.target.checked; render();
  });
  $('togField').addEventListener('change', async (e) => {
    state.viz.showField = e.target.checked;
    if (e.target.checked) await refreshField();
    render();
  });
  $('togTraj').addEventListener('change', (e) => {
    state.viz.showTraj = e.target.checked; render();
  });
  const togLat = $('togLatent');
  if (togLat) {
    togLat.addEventListener('change', async (e) => {
      state.showLatent = e.target.checked;
      if (state.showLatent && isLdm(state.objective)) {
        await ensureLatentCache();
      }
      render();
    });
  }

  $('btnClearDraw').addEventListener('click', () => {
    state.viz.clearDrawing();
    render();
  });

  window.addEventListener('resize', () => {
    state.viz.resize();
    render();
  });
}

async function main() {
  state.viz = new CanvasViz($('mainCanvas'));
  state.lossChart = new LossChart($('lossChart'));
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
  updateLdmPanel();
  setStatus('Ready — hit Train');
  updateMetrics();

  const noise = [];
  for (let i = 0; i < N_GEN; i++) noise.push(randn(), randn());
  state.genFlat = noise;
  render();
}

main().catch((e) => {
  console.error(e);
  setStatus('Failed to start: ' + e.message);
});
