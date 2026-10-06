# DiffLab

**An Interactive, Visual Experimentation Tool for Diffusion Models & Flow Matching**

Inspired by [GAN Lab](https://github.com/poloclub/ganlab) (Polo Club of Data Science).  
DiffLab lets you **train and explore** Denoising Diffusion Probabilistic Models (DDPM), Flow Matching, and velocity / score parameterizations **live in your browser** on 2D toy distributions — no install, no GPU required.

<p align="center">
  <a href="https://karimiannima.github.io/DiffLab/"><strong>🚀 Live Demo</strong></a>
  &nbsp;·&nbsp;
  <a href="#features"><strong>Features</strong></a>
  &nbsp;·&nbsp;
  <a href="#concepts"><strong>Concepts</strong></a>
  &nbsp;·&nbsp;
  <a href="#run-locally"><strong>Run locally</strong></a>
</p>

---

## Overview

Modern generative models (Stable Diffusion, FLUX, SD3, video models, robot policies, …) are built on **diffusion** or **flow matching**.  
They look different on paper, but under Gaussian paths they are two sides of the same coin:

| Framework | Network predicts | Sampling | Paths |
|-----------|------------------|----------|--------|
| **DDPM** | noise ε (or score ∇log p) | reverse SDE / DDIM | curved (variance-preserving) |
| **Flow Matching** | velocity **v** | ODE (Euler / Heun) | often straight (OT) |
| **v-prediction** | velocity (Karras / EDM style) | ODE | depends on schedule |

DiffLab makes the geometry visible: **forward noising**, **learned vector fields**, **sample trajectories**, and **side-by-side DDPM vs Flow Matching** training on the same 2D data.

Everything runs client-side with **TensorFlow.js** (WebGL). Open a browser and play.

---

## Features

- **Live in-browser training** of small MLPs on 2D point clouds
- **Objectives**
  - DDPM noise prediction (ε)
  - Flow Matching / Conditional Flow Matching (velocity **v**)
  - Score / denoising score matching
  - x₀ prediction
- **Datasets**: rings, moons, spiral, Swiss roll, checker, Gaussian mixture, **draw your own**
- **Visualizations**
  - Data vs generated samples
  - Forward diffusion animation
  - Reverse / flow sampling animation
  - Vector field overlay
  - Sample trajectories
  - Loss curve
- **Controls**: learning rate, batch size, timesteps T, noise schedule (linear / cosine), network width/depth, training speed
- **Compare modes**: switch objectives without reloading; reset / reseed
- **Pure static site** — works on GitHub Pages

---

## Concepts covered

1. **Forward process** \( q(x_t \mid x_0) \) — gradual Gaussian noising  
2. **DDPM reverse process** — predict ε, step back with schedule  
3. **Probability-flow ODE / DDIM** — deterministic sibling of DDPM  
4. **Flow Matching** — regress velocity field along a chosen path (linear OT path)  
5. **Equivalence** — Gaussian FM ↔ diffusion under reparameterization (ε ↔ v ↔ score)  
6. **Noise schedules** — linear β vs cosine  
7. **Straight vs curved paths** — why FM often needs fewer steps  

Educational companion reading:
- [Diffusion Meets Flow Matching](https://diffusionflow.github.io/)
- [Diffusion Explorer](https://github.com/helblazer811/Diffusion-Explorer) (Polo Chau lab)
- [GAN Lab](https://github.com/poloclub/ganlab)

---

## Run locally

### Option A — zero build (recommended)

```bash
git clone https://github.com/karimiannima/DiffLab.git
cd DiffLab
python -m http.server 8080
# open http://localhost:8080
```

Or with Node:

```bash
npx serve .
```

### Option B — npm scripts

```bash
npm install
npm start          # static server on :8080
```

### GitHub Pages

Settings → Pages → Deploy from branch `main` / root (or `/docs`).  
The demo is a static `index.html` + ES modules.

---

## Project structure

```
DiffLab/
├── index.html              # App shell
├── css/styles.css          # UI
├── src/
│   ├── main.js             # Bootstrap & training loop
│   ├── models/
│   │   ├── network.js      # MLP with time embedding (TF.js)
│   │   ├── ddpm.js         # DDPM / ε-prediction + sampling
│   │   ├── flow_matching.js# Conditional Flow Matching (velocity)
│   │   └── schedules.js    # β / ᾱ / cosine schedules
│   ├── data/
│   │   └── datasets.js     # Toy 2D distributions + freehand draw
│   ├── viz/
│   │   ├── canvas_viz.js   # Points, trajectories, vector field
│   │   └── charts.js       # Loss chart
│   └── ui/
│       └── controls.js     # Panels & event wiring
├── LICENSE
└── README.md
```

---

## How to use the demo

1. Pick a **dataset** (or draw one).  
2. Choose an **objective**: DDPM (ε), Flow Matching (v), Score, or x₀.  
3. Hit **Train**. Watch loss and the generated cloud morph toward the data.  
4. Toggle **vector field** and **trajectories** to see geometry.  
5. Use **Sample** / **Forward** to scrub the generative process.  
6. Switch objective and retrain — same data, different geometry.

---

## Citation / inspiration

If you use DiffLab in teaching or research, please also cite the works that inspired it:

```
@article{kahng2019ganlab,
  title={GAN Lab: Understanding Complex Deep Generative Models using Interactive Visual Experimentation},
  author={Kahng, Minsuk and Thorat, Nikhil and Chau, Duen Horng and Viégas, Fernanda and Wattenberg, Martin},
  journal={IEEE TVCG},
  year={2019}
}

@article{helbling2025diffusionexplorer,
  title={Diffusion Explorer: Interactive Exploration of Diffusion Models},
  author={Helbling, Alec and Chau, Duen Horng},
  journal={arXiv:2507.01178},
  year={2025}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Nima Karimian** — [github.com/karimiannima](https://github.com/karimiannima)

Built as an open educational companion to GAN Lab for the diffusion / flow-matching era.
