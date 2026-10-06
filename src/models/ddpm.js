/**
 * DDPM training (ε-prediction) and ancestral sampling helpers.
 */

import { predict } from './network.js';

/** Proper training step that applies optimizer gradients. */
export async function ddpmTrainStep(tf, model, optimizer, x0Data, schedule, objective, tDim) {
  const B = x0Data.length / 2;
  const T = schedule.T;

  const { value, grads } = tf.variableGrads(() => {
    const x0 = tf.tensor2d(x0Data, [B, 2]);
    const tIndices = tf.randomUniform([B], 0, T, 'int32');
    const noise = tf.randomNormal([B, 2]);
    const sqrtA = tf.tensor1d(schedule.sqrtAlphasCumprod);
    const sqrtOm = tf.tensor1d(schedule.sqrtOneMinusAlphasCumprod);
    const sa = tf.gather(sqrtA, tIndices).expandDims(1);
    const so = tf.gather(sqrtOm, tIndices).expandDims(1);
    const xt = x0.mul(sa).add(noise.mul(so));
    const tCont = tIndices.toFloat().add(1).div(T);
    const pred = predict(tf, model, xt, tCont, tDim);

    let target;
    if (objective === 'x0') {
      target = x0;
    } else if (objective === 'score') {
      target = noise.div(so.add(1e-5)).mul(-1);
    } else {
      target = noise;
    }
    return pred.sub(target).square().mean();
  });

  optimizer.applyGradients(grads);
  Object.values(grads).forEach((g) => g.dispose && g.dispose());
  const lossNum = await value.data();
  value.dispose();
  return lossNum[0];
}

/**
 * Ancestral DDPM sampling (one step).
 */
export function ddpmPSampleStep(tf, model, xt, t, schedule, objective, tDim) {
  return tf.tidy(() => {
    const T = schedule.T;
    const tIdx = Math.max(0, Math.min(T - 1, t | 0));
    const beta = schedule.betas[tIdx];
    const alpha = schedule.alphas[tIdx];
    const alphabar = schedule.alphasCumprod[tIdx];
    const alphabarPrev = tIdx > 0 ? schedule.alphasCumprod[tIdx - 1] : 1;

    const tCont = tf.fill([xt.shape[0]]).add((tIdx + 1) / T);
    const pred = predict(tf, model, xt, tCont, tDim);

    let eps, x0pred;
    const sqrtAb = Math.sqrt(alphabar);
    const sqrtOm = Math.sqrt(1 - alphabar);

    if (objective === 'eps') {
      eps = pred;
      x0pred = xt.sub(eps.mul(sqrtOm)).div(sqrtAb + 1e-8);
    } else if (objective === 'x0') {
      x0pred = pred;
      eps = xt.sub(x0pred.mul(sqrtAb)).div(sqrtOm + 1e-8);
    } else if (objective === 'score') {
      eps = pred.mul(-sqrtOm);
      x0pred = xt.sub(eps.mul(sqrtOm)).div(sqrtAb + 1e-8);
    } else {
      eps = pred;
      x0pred = xt.sub(eps.mul(sqrtOm)).div(sqrtAb + 1e-8);
    }

    const coefX0 = (Math.sqrt(alphabarPrev) * beta) / (1 - alphabar + 1e-8);
    const coefXt = (Math.sqrt(alpha) * (1 - alphabarPrev)) / (1 - alphabar + 1e-8);
    const mean = x0pred.mul(coefX0).add(xt.mul(coefXt));

    if (tIdx === 0) return mean;

    const noise = tf.randomNormal(xt.shape);
    const sigma = Math.sqrt(beta);
    return mean.add(noise.mul(sigma));
  });
}

/** Full sampling loop → number[] [N*2] */
export async function sampleDDPM(tf, model, n, schedule, objective, tDim, steps = null) {
  const T = schedule.T;
  const useSteps = steps || T;
  let xt = tf.randomNormal([n, 2]);
  const stride = Math.max(1, Math.floor(T / useSteps));

  for (let t = T - 1; t >= 0; t -= stride) {
    const next = ddpmPSampleStep(tf, model, xt, t, schedule, objective, tDim);
    xt.dispose();
    xt = next;
  }
  const data = await xt.data();
  xt.dispose();
  return Array.from(data);
}

/**
 * Predict vector field for visualization.
 */
export function fieldFromModel(tf, model, xs, ys, tCont, objective, schedule, tDim) {
  return tf.tidy(() => {
    const xy = tf.tensor2d(
      xs.map((x, i) => [x, ys[i]]),
      [xs.length, 2]
    );
    const t = tf.fill([xs.length]).add(tCont);
    const pred = predict(tf, model, xy, t, tDim);

    if (objective === 'score') return pred;
    if (objective === 'x0') return pred.sub(xy);
    return pred.mul(-1);
  });
}
