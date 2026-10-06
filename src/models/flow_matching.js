/**
 * Conditional Flow Matching (OT / linear path).
 * x_t = (1-t) * noise + t * data
 * v_target = data - noise
 */

import { predict } from './network.js';

export async function fmTrainStep(tf, model, optimizer, xDataFlat, tDim) {
  const B = xDataFlat.length / 2;

  const { value, grads } = tf.variableGrads(() => {
    const x1 = tf.tensor2d(xDataFlat, [B, 2]);
    const x0 = tf.randomNormal([B, 2]);
    const t = tf.randomUniform([B]);
    const tExp = t.expandDims(1);
    const xt = x0.mul(tExp.mul(-1).add(1)).add(x1.mul(tExp));
    const vTarget = x1.sub(x0);
    const pred = predict(tf, model, xt, t, tDim);
    return pred.sub(vTarget).square().mean();
  });

  optimizer.applyGradients(grads);
  Object.values(grads).forEach((g) => g.dispose && g.dispose());
  const lossNum = await value.data();
  value.dispose();
  return lossNum[0];
}

/** Euler ODE sampling: dx/dt = v_θ(x,t), t: 0 → 1 */
export async function sampleFM(tf, model, n, tDim, steps = 50) {
  let x = tf.randomNormal([n, 2]);
  const dt = 1 / steps;

  for (let i = 0; i < steps; i++) {
    const tVal = i / steps;
    const next = tf.tidy(() => {
      const t = tf.fill([n]).add(tVal);
      const v = predict(tf, model, x, t, tDim);
      return x.add(v.mul(dt));
    });
    x.dispose();
    x = next;
  }
  const data = await x.data();
  x.dispose();
  return Array.from(data);
}

export async function sampleFMAt(tf, model, n, tDim, tEnd, steps = 40) {
  let x = tf.randomNormal([n, 2]);
  const nSteps = Math.max(1, Math.round(steps * tEnd));
  const dt = tEnd / nSteps;

  for (let i = 0; i < nSteps; i++) {
    const tVal = (i / nSteps) * tEnd;
    const next = tf.tidy(() => {
      const t = tf.fill([n]).add(tVal);
      const v = predict(tf, model, x, t, tDim);
      return x.add(v.mul(dt));
    });
    x.dispose();
    x = next;
  }
  const data = await x.data();
  x.dispose();
  return Array.from(data);
}

export function fmField(tf, model, xs, ys, tCont, tDim) {
  return tf.tidy(() => {
    const xy = tf.tensor2d(
      xs.map((x, i) => [x, ys[i]]),
      [xs.length, 2]
    );
    const t = tf.fill([xs.length]).add(tCont);
    return predict(tf, model, xy, t, tDim);
  });
}
