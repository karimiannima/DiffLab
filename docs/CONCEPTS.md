# DiffLab Concepts

## 1. DDPM (Denoising Diffusion Probabilistic Models)

Forward (fixed):
\[
q(x_t \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar\alpha_t}\, x_0,\, (1-\bar\alpha_t)I\right)
\]

Training targets: **ε-prediction**, **x₀-prediction**, or **score**.

---

## 2. Flow Matching

Linear OT path from noise \(x_0=\varepsilon\) to data \(x_1\):
\[
x_t = (1-t)\,\varepsilon + t\, x_1, \qquad v^\star = x_1 - \varepsilon
\]
\[
\mathcal{L} = \mathbb{E}\,\|v_\theta(x_t,t) - v^\star\|^2
\]
Sample with ODE \(\mathrm{d}x/\mathrm{d}t = v_\theta\).

---

## 3. Score-based Generative Models (Song et al.)

Estimate the **Stein score** \( \nabla_x \log p_t(x) \) with **denoising score matching (DSM)**.

### VE-SDE (variance exploding)
\[
x_t = x_0 + \sigma(t)\,\varepsilon, \quad
\sigma(t)=\sigma_{\min}\big(\sigma_{\max}/\sigma_{\min}\big)^t
\]
Target: \( s^\star = -\varepsilon / \sigma(t) \).

### VP-SDE (variance preserving)
Continuous analogue of DDPM marginals:
\[
x_t = \sqrt{\bar\alpha(t)}\, x_0 + \sqrt{1-\bar\alpha(t)}\,\varepsilon
\]
Target: \( s^\star = -\varepsilon / \sqrt{1-\bar\alpha(t)} \).

Sampling uses the **reverse SDE** (Euler–Maruyama) and optional **Langevin correctors**.  
Score ↔ noise: \( \varepsilon \approx -\sigma\, s \).

---

## 4. Latent Diffusion Models (LDMs)

High-res image models (Stable Diffusion) diffuse in a **compressed latent**:

```
pixel/data x  →  Encoder E  →  latent z
                     ↓
              Diffusion / FM on z
                     ↓
latent ẑ  →  Decoder D  →  reconstruction x̂
```

**DiffLab toy LDM**
1. Train MLP autoencoder \(E, D\) with reconstruction (+ mild latent reg).
2. Train DDPM (**ε**) or Flow Matching (**v**) on \(z = E(x)\).
3. Sample \(z\) from noise, then \(x = D(z)\).

Toggle **Show latent space** to plot \(z\) instead of \(x\).

Why latents?
- Lower dimension / smoother manifold → cheaper denoising  
- Perceptual compression keeps semantics in \(z\)  
- Same recipe scales to images (VAE/VQGAN + U-Net/DiT)

---

## 5. How the pieces relate

| View | Network output | Path |
|------|----------------|------|
| DDPM | ε or x₀ | discrete VP |
| Score SDE | ∇log p | VE or VP continuous |
| Flow Matching | velocity v | often straight OT |
| LDM | any of the above **in z** | + autoencoder |

Under Gaussian paths many of these are **reparameterizations** of the same generative process.

---

## References
- Ho et al., DDPM (2020)
- Song et al., Score SDE (ICLR 2021)
- Lipman et al., Flow Matching (ICLR 2023)
- Rombach et al., Latent Diffusion / Stable Diffusion (CVPR 2022)
- Kahng et al., GAN Lab (IEEE TVCG 2019)
