/**
 * Noise / probability path schedules for DDPM and Flow Matching.
 */

export function linearBetaSchedule(T, betaStart = 1e-4, betaEnd = 0.02) {
  const betas = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    betas[t] = betaStart + (betaEnd - betaStart) * (t / Math.max(T - 1, 1));
  }
  return betas;
}

/** Nichol & Dhariwal cosine schedule (clamped). */
export function cosineBetaSchedule(T, s = 0.008) {
  const alphasCumprod = new Float32Array(T);
  const f = (t) => Math.cos(((t / T + s) / (1 + s)) * (Math.PI / 2)) ** 2;
  const f0 = f(0);
  for (let t = 0; t < T; t++) {
    alphasCumprod[t] = Math.min(f(t + 1) / f0, 0.9999);
  }
  const betas = new Float32Array(T);
  betas[0] = 1 - alphasCumprod[0];
  for (let t = 1; t < T; t++) {
    betas[t] = 1 - alphasCumprod[t] / alphasCumprod[t - 1];
    betas[t] = Math.min(Math.max(betas[t], 1e-5), 0.999);
  }
  return betas;
}

export function buildSchedule(T, type = 'linear') {
  const betas = type === 'cosine' ? cosineBetaSchedule(T) : linearBetaSchedule(T);
  const alphas = new Float32Array(T);
  const alphasCumprod = new Float32Array(T);
  let cum = 1;
  for (let t = 0; t < T; t++) {
    alphas[t] = 1 - betas[t];
    cum *= alphas[t];
    alphasCumprod[t] = cum;
  }
  // sqrt terms for q-sample
  const sqrtAlphasCumprod = new Float32Array(T);
  const sqrtOneMinusAlphasCumprod = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    sqrtAlphasCumprod[t] = Math.sqrt(alphasCumprod[t]);
    sqrtOneMinusAlphasCumprod[t] = Math.sqrt(1 - alphasCumprod[t]);
  }
  return {
    T,
    type,
    betas,
    alphas,
    alphasCumprod,
    sqrtAlphasCumprod,
    sqrtOneMinusAlphasCumprod,
  };
}

/** Continuous-time linear interpolant for flow matching: x_t = (1-t) x0 + t x1  (here x0=noise, x1=data) or vice versa.
 *  We use x_t = (1-t)*noise + t*data  so t=0 is noise, t=1 is data (common FM convention).
 *  Velocity target: v = data - noise.
 */
export function fmLinearPath(xNoise, xData, t) {
  // x_t = (1-t) noise + t data
  const xt = xNoise.map((n, i) => (1 - t) * n + t * xData[i]);
  const v = xData.map((d, i) => d - xNoise[i]);
  return { xt, v };
}

export function tContinuous(discreteT, T) {
  // map discrete step index [0..T-1] to continuous t in (0,1]
  return (discreteT + 1) / T;
}
