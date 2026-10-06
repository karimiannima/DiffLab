# DiffLab

**An Interactive, Visual Experimentation Tool for Diffusion, Score-based Models & Latent Diffusion**


> ## ⚠️ Demo URL 404?
>
> **`https://karimiannima.github.io/DiffLab/` only works if this repository is Public and Pages is enabled.**
>
> 1. **Settings → General → Danger zone → Change visibility → Public**
> 2. **Settings → Pages → Source → `Deploy from a branch` → `main` / `/ (root)` → Save**
>    *(or Source → GitHub Actions, then run the “Deploy DiffLab to GitHub Pages” workflow)*
> 3. Wait 1–2 minutes, hard-refresh the demo link.
>
> Until the repo is **Public**, GitHub returns **404 File not found** for both the repo page and Pages (this is expected for private repos on free accounts).

Train and explore **DDPM**, **Flow Matching**, **Score-based Generative Models (VE/VP SDE)**, and **Latent Diffusion Models (LDMs)** live in your browser on 2D distributions.

<p align="center">
  <a href="https://raw.githack.com/karimiannima/DiffLab/main/index.html"><strong>🚀 Live Demo (works now)</strong></a>
  &nbsp;·&nbsp;
  <a href="https://karimiannima.github.io/DiffLab/"><strong>GitHub Pages</strong></a>
  &nbsp;·&nbsp;
  <a href="docs/CONCEPTS.md"><strong>Concepts</strong></a>
  &nbsp;·&nbsp;
  <a href="#run-locally"><strong>Run locally</strong></a>
</p>

---

## What’s inside (v1.1)

| Family | What you train | Sampling |
|--------|----------------|----------|
| **Flow Matching** | velocity **v** on linear OT path | ODE (Euler) |
| **DDPM** | noise **ε** or **x₀** | ancestral reverse chain |
| **Score DSM** | score ∇log p via denoising score matching | DDPM-style |
| **Score VE-SDE** | continuous score, variance-exploding SDE (Song et al.) | reverse SDE + Langevin |
| **Score VP-SDE** | continuous score, variance-preserving SDE | reverse SDE |
| **LDM · latent DDPM** | autoencoder **E/D** + **ε**-diffusion in **z** | denoise **z**, decode **x=D(z)** |
| **LDM · latent FM** | autoencoder + velocity in latent | ODE in **z**, decode |

### Visual tools
- Data vs generated clouds
- **Vector / score fields**
- Sample **trajectories**
- Forward noising (data space or **latent** for LDM)
- Reverse generation animation
- **Show latent space (z)** toggle for LDMs
- Freehand **draw-your-own** distribution
- Loss chart, schedules (linear / cosine)

Everything runs **client-side** with **TensorFlow.js** (WebGL). No server GPU required.

---

## Concepts at a glance

### Score-based generative models
Learn \( s_\theta(x,t) \approx \nabla_x \log p_t(x) \) with denoising score matching, then sample with reverse SDEs (VE / VP) or probability-flow ODEs. DiffLab exposes **VE-SDE**, **VP-SDE**, and discrete **DSM**.

### Latent Diffusion Models (LDMs)
As in **Stable Diffusion** (Rombach et al.):

```
x  --E-->  z  --diffuse/FM-->  ẑ  --D-->  x̂
```

DiffLab trains a tiny MLP autoencoder jointly with a latent denoiser (DDPM or Flow Matching). Toggle **Show latent space** to inspect the **z** cloud.

### Equivalence
Under Gaussian paths, diffusion, score matching, and flow matching are closely related reparameterizations (ε ↔ score ↔ v). See [diffusionflow.github.io](https://diffusionflow.github.io/) and `docs/CONCEPTS.md`.

---

## Run locally

```bash
git clone https://github.com/karimiannima/DiffLab.git
cd DiffLab
python -m http.server 8080
# open http://localhost:8080
```

Or: `npm start` / `npx serve .`

### GitHub Pages

> **404 on the demo link?** See [docs/PAGES.md](docs/PAGES.md) — you must set **Settings → Pages → Source** once (GitHub Actions or branch `main` / root).
Settings → Pages → branch `main` / root →
https://karimiannima.github.io/DiffLab/

---

## Project structure

```
DiffLab/
├── index.html
├── css/styles.css
├── src/
│   ├── main.js
│   ├── models/
│   │   ├── network.js          # time-conditioned MLP
│   │   ├── schedules.js        # β / cosine / FM path
│   │   ├── ddpm.js             # ε / x₀ / DSM + sampling
│   │   ├── flow_matching.js    # velocity CFM
│   │   ├── score_sde.js        # VE / VP score SDEs
│   │   └── ldm.js              # AE + latent DDPM/FM
│   ├── data/datasets.js
│   └── viz/                    # canvas + loss chart
└── docs/CONCEPTS.md
```

---

## Citation / inspiration

```
}
@inproceedings{song2021score,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Song, Yang and others}, booktitle={ICLR}, year={2021}
}
@inproceedings{rombach2022ldm,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and others}, booktitle={CVPR}, year={2022}
}
@article{helbling2025diffusionexplorer,
  title={Diffusion Explorer: Interactive Exploration of Diffusion Models},
  author={Helbling, Alec and Chau, Duen Horng},
  journal={arXiv:2507.01178}, year={2025}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

**Author:** [Nima Karimian](https://github.com/karimiannima)
