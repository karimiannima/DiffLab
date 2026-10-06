/**
 * Score-based Generative Models (Song et al.)
 * VE-SDE / VP-SDE + denoising score matching + reverse sampling
 */

import { predict } from './network.js';

export function veSigma(t, sigmaMin = 0.01, sigmaMax = 2.0) {
  return sigmaMin * Math.pow(sigmaMax / sigmaMin, t);
}

export function vpMarginal(t, betaMin = 0.1, betaMax = 20) {
  const integral = 0.5 * t * t * (betaMax - betaMin) + t * betaMin;
  const alphaBar = Math.exp(-integral);
  return {
    alphaBar,
    std: Math.sqrt(Math.max(1 - alphaBar, 1e-8)),
    meanScale: Math.sqrt(alphaBar),
  };
}

/**
 * DSM train step. Builds continuous noise levels outside the tape via sampling t first.
 */
export async function scoreTrainStep(tf, model, optimizer, x0Data, tDim, sdeType = 've') {
  const B = x0Data.length / 2;

  // Sample t and noise levels in JS so the tape only sees tensors
  const tArr = new Float32Array(B);
  const meanArr = new Float32Array(B);
  const stdArr = new Float32Array(B);
  for (let i = 0; i < B; i++) {
    const t = Math.random();
    tArr[i] = t;
    if (sdeType === 'vp') {
      const m = vpMarginal(Math.max(t, 1e-4));
      meanArr[i] = m.meanScale;
      stdArr[i] = m.std;
    } else {
      meanArr[i] = 1;
      stdArr[i] = veSigma(Math.max(t, 1e-4));
    }
  }

  const { value, grads } = tf.variableGrads(() => {
    const x0 = tf.tensor2d(x0Data, [B, 2]);
    const t = tf.tensor1d(tArr);
    const noise = tf.randomNormal([B, 2]);
    const meanT = tf.tensor1d(meanArr).expandDims(1);
    const stdT = tf.tensor1d(stdArr).expandDims(1);

    // VE: x = x0 + σ ε ; VP: x = √ā x0 + σ ε
    const xt = sdeType === 'vp'
      ? x0.mul(meanT).add(noise.mul(stdT))
      : x0.add(noise.mul(stdT));

    const target = noise.div(stdT.add(1e-5)).mul(-1);
    const weight = stdT.square();
    const pred = predict(tf, model, xt, t, tDim);
    return pred.sub(target).square().mul(weight).mean();
  });

  optimizer.applyGradients(grads);
  Object.values(grads).forEach((g) => g.dispose && g.dispose());
  const lossNum = await value.data();
  value.dispose();
  return lossNum[0];
}

export async function sampleScoreSDE(tf, model, n, tDim, sdeType = 've', steps = 50) {
  let x = tf.randomNormal([n, 2]).mul(sdeType === 've' ? 2.0 : 1.0);

  for (let i = steps - 1; i >= 0; i--) {
    const tVal = (i + 1) / steps;
    const tNext = i / steps;

    const next = tf.tidy(() => {
      const t = tf.fill([n]).add(tVal);
      const score = predict(tf, model, x, t, tDim);

      if (sdeType === 've') {
        const sig = veSigma(Math.max(tVal, 1e-4));
        const sigNext = veSigma(Math.max(tNext, 1e-4));
        const dSig2 = sig * sig - sigNext * sigNext;
        const drift = score.mul(dSig2);
        const noise = tf.randomNormal(x.shape).mul(Math.sqrt(Math.max(dSig2, 0)));
        const eps = 0.15 * (sig * sig);
        const corr = score.mul(eps).add(tf.randomNormal(x.shape).mul(Math.sqrt(2 * eps)));
        return x.add(drift).add(corr.mul(0.5));
      }

      const m = vpMarginal(tVal);
      const eps = score.mul(-m.std);
      const x0 = x.sub(eps.mul(m.std)).div(m.meanScale + 1e-8);
      const direction = x0.sub(x);
      const noise = i > 0 ? tf.randomNormal(x.shape).mul(0.15) : tf.zeros(x.shape);
      return x.add(direction.mul(1 / steps + 0.12)).add(noise);
    });

    x.dispose();
    x = next;
  }

  const data = await x.data();
  x.dispose();
  return Array.from(data);
}

export function scoreField(tf, model, xs, ys, tCont, tDim) {
  return tf.tidy(() => {
    const xy = tf.tensor2d(xs.map((x, i) => [x, ys[i]]), [xs.length, 2]);
    const t = tf.fill([xs.length]).add(tCont);
    return predict(tf, model, xy, t, tDim);
  });
}
