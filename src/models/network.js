/**
 * Small MLP with sinusoidal time embedding — TensorFlow.js
 * Input: (x, y, t_emb...) -> (out_x, out_y)
 */

export function createTimeMLP({
  tf,
  hidden = 128,
  depth = 3,
  tDim = 16,
  outDim = 2,
} = {}) {
  const layers = [];
  // Input: 2 (xy) + tDim
  let inDim = 2 + tDim;

  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: hidden,
    activation: 'elu',
    inputShape: [inDim],
    kernelInitializer: 'heNormal',
  }));
  for (let i = 1; i < depth; i++) {
    model.add(tf.layers.dense({
      units: hidden,
      activation: 'elu',
      kernelInitializer: 'heNormal',
    }));
  }
  model.add(tf.layers.dense({
    units: outDim,
    kernelInitializer: 'zeros',
  }));
  return model;
}

/** Sinusoidal embedding for continuous t in [0,1] or discrete steps. */
export function timeEmbedding(tf, t, dim = 16) {
  // t: Tensor1d batch
  return tf.tidy(() => {
    const half = dim / 2;
    const freqs = tf.exp(
      tf.linspace(0, Math.log(10000), half).mul(-1).div(half - 1 || 1)
    );
    // t[:, None] * freqs[None, :]
    const args = t.expandDims(1).mul(freqs.expandDims(0)).mul(2 * Math.PI);
    const sin = args.sin();
    const cos = args.cos();
    return tf.concat([sin, cos], 1);
  });
}

export function predict(tf, model, xy, t, tDim = 16) {
  return tf.tidy(() => {
    const emb = timeEmbedding(tf, t, tDim);
    const inp = tf.concat([xy, emb], 1);
    return model.predict(inp);
  });
}
