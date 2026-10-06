# DiffLab Concepts

## DDPM (Denoising Diffusion Probabilistic Models)

Forward (fixed):
\[
q(x_t \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar\alpha_t}\, x_0,\, (1-\bar\alpha_t)I\right)
\]

Training: predict noise \(\varepsilon\) added at random \(t\):
\[
\mathcal{L} = \mathbb{E}\,\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2
\]

Sampling: reverse the chain (ancestral or DDIM).

## Flow Matching

Choose a path from noise \(x_0=\varepsilon\) to data \(x_1\). Linear OT path:
\[
x_t = (1-t)\,\varepsilon + t\, x_1, \qquad v^\star = x_1 - \varepsilon
\]

Train by regression:
\[
\mathcal{L} = \mathbb{E}\,\|v_\theta(x_t,t) - v^\star\|^2
\]

Sample by integrating the ODE \(\mathrm{d}x/\mathrm{d}t = v_\theta(x,t)\) from \(t=0\) to \(1\).

## Equivalence

Under Gaussian paths, diffusion and flow matching describe the same family of models; **ε-prediction**, **x₀-prediction**, **v-prediction**, and **score** are reparameterizations. See [diffusionflow.github.io](https://diffusionflow.github.io/).

## Why 2D?

Like GAN Lab, low-dimensional data makes **vector fields**, **trajectories**, and **mode covering** visible — the same geometry that runs latent image/video generators.
