/**
 * Toy Latent Diffusion Model (LDM) for 2D DiffLab.
 *
 * Pipeline (mirrors Stable Diffusion / Rombach et al.):
 *   1. Autoencoder: x --E--> z --D--> x̂   (perceptual compression)
 *   2. Diffusion / Flow Matching trained in latent z-space
 *   3. Sample: z_T ~ N(0,I) → denoise to z_0 → x = D(z_0)
 *
 * Here encoder/decoder are small MLPs (2D→latentDim→2D). Latent dim
 * defaults to 2 so we can still plot z-space geometry; set latentDim=1
 * for a true bottleneck demo.
 */

import { predict, createTimeMLP } from './network.js';
import { ddpmTrainStep, sampleDDPM } from './ddpm.js';
import { fmTrainStep, sampleFM } from './flow_matching.js';

export function createAutoencoder(tf, { latentDim = 2, hidden = 64 } = {}) {
  const encoder = tf.sequential();
  encoder.add(tf.layers.dense({
    units: hidden, activation: 'elu', inputShape: [2], kernelInitializer: 'heNormal',
  }));
  encoder.add(tf.layers.dense({
    units: hidden, activation: 'elu', kernelInitializer: 'heNormal',
  }));
  encoder.add(tf.layers.dense({
    units: latentDim, kernelInitializer: 'heNormal',
  }));

  const decoder = tf.sequential();
  decoder.add(tf.layers.dense({
    units: hidden, activation: 'elu', inputShape: [latentDim], kernelInitializer: 'heNormal',
  }));
  decoder.add(tf.layers.dense({
    units: hidden, activation: 'elu', kernelInitializer: 'heNormal',
  }));
  decoder.add(tf.layers.dense({
    units: 2, kernelInitializer: 'heNormal',
  }));

  return { encoder, decoder, latentDim };
}

export async function aeTrainStep(tf, encoder, decoder, optimizer, xDataFlat) {
  const B = xDataFlat.length / 2;
  const { value, grads } = tf.variableGrads(() => {
    const x = tf.tensor2d(xDataFlat, [B, 2]);
    const z = encoder.predict(x);
    const xHat = decoder.predict(z);
    // recon + mild latent reg (keep z ~ N(0,1) scale)
    const recon = xHat.sub(x).square().mean();
    const latReg = z.square().mean().mul(0.001);
    return recon.add(latReg);
  });
  optimizer.applyGradients(grads);
  Object.values(grads).forEach((g) => g.dispose && g.dispose());
  const v = await value.data();
  value.dispose();
  return v[0];
}

/** Encode data flat array → latent flat array */
export async function encodeFlat(tf, encoder, dataFlat) {
  const n = dataFlat.length / 2;
  const x = tf.tensor2d(dataFlat, [n, 2]);
  const z = encoder.predict(x);
  const d = await z.data();
  x.dispose();
  z.dispose();
  return Array.from(d);
}

export async function decodeFlat(tf, decoder, zFlat, latentDim) {
  const n = zFlat.length / latentDim;
  const z = tf.tensor2d(zFlat, [n, latentDim]);
  const x = decoder.predict(z);
  const d = await x.data();
  z.dispose();
  x.dispose();
  return Array.from(d);
}

/**
 * Create diffusion model that operates on latentDim-dimensional inputs.
 * Reuses time-MLP with input dim latentDim instead of 2.
 */
export function createLatentDiffModel(tf, { latentDim = 2, hidden = 128, depth = 3, tDim = 16 } = {}) {
  const model = tf.sequential();
  const inDim = latentDim + tDim;
  model.add(tf.layers.dense({
    units: hidden, activation: 'elu', inputShape: [inDim], kernelInitializer: 'heNormal',
  }));
  for (let i = 1; i < depth; i++) {
    model.add(tf.layers.dense({
      units: hidden, activation: 'elu', kernelInitializer: 'heNormal',
    }));
  }
  model.add(tf.layers.dense({ units: latentDim, kernelInitializer: 'zeros' }));
  return model;
}

export function predictLatent(tf, model, z, t, tDim, latentDim) {
  return tf.tidy(() => {
    // time embedding inline (same as network.timeEmbedding)
    const half = tDim / 2;
    const freqs = tf.exp(
      tf.linspace(0, Math.log(10000), half).mul(-1).div(half - 1 || 1)
    );
    const args = t.expandDims(1).mul(freqs.expandDims(0)).mul(2 * Math.PI);
    const emb = tf.concat([args.sin(), args.cos()], 1);
    const inp = tf.concat([z, emb], 1);
    return model.predict(inp);
  });
}

/** Train latent diffusion (ε or FM) on encoded batch */
export async function ldmDiffTrainStep(tf, model, optimizer, zFlat, latentDim, schedule, mode, tDim) {
  // mode: 'eps' | 'fm'
  if (mode === 'fm') {
    // adapt FM for latent dim
    const B = zFlat.length / latentDim;
    const { value, grads } = tf.variableGrads(() => {
      const x1 = tf.tensor2d(zFlat, [B, latentDim]);
      const x0 = tf.randomNormal([B, latentDim]);
      const t = tf.randomUniform([B]);
      const tExp = t.expandDims(1);
      const xt = x0.mul(tExp.mul(-1).add(1)).add(x1.mul(tExp));
      const vTarget = x1.sub(x0);
      const pred = predictLatent(tf, model, xt, t, tDim, latentDim);
      return pred.sub(vTarget).square().mean();
    });
    optimizer.applyGradients(grads);
    Object.values(grads).forEach((g) => g.dispose && g.dispose());
    const lossNum = await value.data();
    value.dispose();
    return lossNum[0];
  }

  // ε-prediction DDPM in latent
  const B = zFlat.length / latentDim;
  const T = schedule.T;
  const { value, grads } = tf.variableGrads(() => {
    const x0 = tf.tensor2d(zFlat, [B, latentDim]);
    const tIndices = tf.randomUniform([B], 0, T, 'int32');
    const noise = tf.randomNormal([B, latentDim]);
    const sqrtA = tf.tensor1d(schedule.sqrtAlphasCumprod);
    const sqrtOm = tf.tensor1d(schedule.sqrtOneMinusAlphasCumprod);
    const sa = tf.gather(sqrtA, tIndices).expandDims(1);
    const so = tf.gather(sqrtOm, tIndices).expandDims(1);
    const xt = x0.mul(sa).add(noise.mul(so));
    const tCont = tIndices.toFloat().add(1).div(T);
    const pred = predictLatent(tf, model, xt, tCont, tDim, latentDim);
    return pred.sub(noise).square().mean();
  });
  optimizer.applyGradients(grads);
  Object.values(grads).forEach((g) => g.dispose && g.dispose());
  const lossNum = await value.data();
  value.dispose();
  return lossNum[0];
}

export async function sampleLDM(tf, model, decoder, n, latentDim, schedule, mode, tDim, steps) {
  let z;
  if (mode === 'fm') {
    // Euler FM in latent
    z = tf.randomNormal([n, latentDim]);
    const dt = 1 / steps;
    for (let i = 0; i < steps; i++) {
      const tVal = i / steps;
      const next = tf.tidy(() => {
        const t = tf.fill([n]).add(tVal);
        const v = predictLatent(tf, model, z, t, tDim, latentDim);
        return z.add(v.mul(dt));
      });
      z.dispose();
      z = next;
    }
  } else {
    // DDPM ancestral in latent
    z = tf.randomNormal([n, latentDim]);
    const T = schedule.T;
    const stride = Math.max(1, Math.floor(T / steps));
    for (let t = T - 1; t >= 0; t -= stride) {
      const next = tf.tidy(() => {
        const tIdx = t;
        const beta = schedule.betas[tIdx];
        const alpha = schedule.alphas[tIdx];
        const alphabar = schedule.alphasCumprod[tIdx];
        const alphabarPrev = tIdx > 0 ? schedule.alphasCumprod[tIdx - 1] : 1;
        const tCont = tf.fill([n]).add((tIdx + 1) / T);
        const eps = predictLatent(tf, model, z, tCont, tDim, latentDim);
        const sqrtAb = Math.sqrt(alphabar);
        const sqrtOm = Math.sqrt(1 - alphabar);
        const x0pred = z.sub(eps.mul(sqrtOm)).div(sqrtAb + 1e-8);
        const coefX0 = (Math.sqrt(alphabarPrev) * beta) / (1 - alphabar + 1e-8);
        const coefXt = (Math.sqrt(alpha) * (1 - alphabarPrev)) / (1 - alphabar + 1e-8);
        const mean = x0pred.mul(coefX0).add(z.mul(coefXt));
        if (tIdx === 0) return mean;
        return mean.add(tf.randomNormal(z.shape).mul(Math.sqrt(beta)));
      });
      z.dispose();
      z = next;
    }
  }

  const x = decoder.predict(z);
  const data = await x.data();
  const zData = await z.data();
  z.dispose();
  x.dispose();
  return { xFlat: Array.from(data), zFlat: Array.from(zData) };
}
