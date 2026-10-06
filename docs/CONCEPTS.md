# DiffLab Concepts

## 1. DDPM (Denoising Diffusion Probabilistic Models)

Forward (fixed):
\[
q(x_t \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar\alpha_t}\, x_0,\, (1-\bar\alpha_t)I\right)
\]

Training targets: **ε-prediction**, **x₀-prediction**, or **score**.

Key paper: Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models* (NeurIPS 2020) [[arXiv:2006.11239](https://arxiv.org/abs/2006.11239)].

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

Key paper: Lipman et al., *Flow Matching for Generative Modeling* (ICLR 2023) [[arXiv:2210.02747](https://arxiv.org/abs/2210.02747)].  
Also: Liu, Gong & Liu, *Flow Straight and Fast: Rectified Flow* (2022) [[arXiv:2209.03003](https://arxiv.org/abs/2209.03003)].

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

Key paper: Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations* (ICLR 2021) [[arXiv:2011.13456](https://arxiv.org/abs/2011.13456)].  
Earlier: Song & Ermon, *Generative Modeling by Estimating Gradients of the Data Distribution* (NeurIPS 2019) [[arXiv:1907.05600](https://arxiv.org/abs/1907.05600)].

---

## 4. Latent Diffusion Models (LDMs) & Stable Diffusion

High-res image generators (including **Stable Diffusion**) run the diffusion process in a **compressed latent space** instead of pixel space:

```
pixel image x  →  Encoder E (VAE)  →  latent z
                         ↓
              U-Net / DiT denoiser on z  (+ text condition)
                         ↓
latent ẑ  →  Decoder D  →  image x̂
```

### Why this matters (Stable Diffusion stack)

| Component | Role in Stable Diffusion |
|-----------|---------------------------|
| **VAE encoder/decoder** | Perceptual compression: \(x \leftrightarrow z\) at lower spatial resolution |
| **Latent denoiser (U-Net)** | Predicts noise (or velocity) on \(z_t\), conditioned on timestep |
| **CLIP / text encoder** | Turns prompts into embeddings for **cross-attention** in the U-Net |
| **Classifier-free guidance (CFG)** | Mixes conditional & unconditional predictions for stronger prompt adherence |
| **Scheduler** | Discrete DDPM/DDIM (or later Euler / flow) steps in latent space |

**DiffLab toy LDM** mirrors the same pipeline on 2D points:
1. Train MLP autoencoder \(E, D\) with reconstruction (+ mild latent reg).
2. Train DDPM (**ε**) or Flow Matching (**v**) on \(z = E(x)\).
3. Sample \(z\) from noise, then \(x = D(z)\).

Toggle **Show latent space** to plot \(z\) instead of \(x\).

Why latents?
- Lower dimension / smoother manifold → cheaper denoising  
- Perceptual compression keeps semantics in \(z\)  
- Same recipe scales to images (VAE + U-Net/DiT)

### Stable Diffusion lineage (short)

1. **LDM paper** (Rombach et al., CVPR 2022) — latent diffusion + cross-attention conditioning  
2. **Stable Diffusion 1.x** — open release of an LDM trained on LAION, with CLIP text encoder  
3. **SD 2.x / SDXL** — larger backbones, better VAE, refined training  
4. **SD3 / FLUX-style** — often switch training objective toward **flow matching / rectified flow** in latent space (same “encode → generate in \(z\) → decode” idea)

---

## 5. How the pieces relate

| View | Network output | Path |
|------|----------------|------|
| DDPM | ε or x₀ | discrete VP |
| Score SDE | ∇log p | VE or VP continuous |
| Flow Matching | velocity v | often straight OT |
| LDM / Stable Diffusion | any of the above **in z** | + VAE + (optional) text condition |

Under Gaussian paths many of these are **reparameterizations** of the same generative process.

---

## References

### Core diffusion & score matching
1. **Ho, J., Jain, A., & Abbeel, P.** (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.  
   https://arxiv.org/abs/2006.11239  
2. **Song, J., Meng, C., & Ermon, S.** (2021). *Denoising Diffusion Implicit Models* (DDIM). ICLR.  
   https://arxiv.org/abs/2010.02502  
3. **Song, Y., & Ermon, S.** (2019). *Generative Modeling by Estimating Gradients of the Data Distribution*. NeurIPS.  
   https://arxiv.org/abs/1907.05600  
4. **Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B.** (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR.  
   https://arxiv.org/abs/2011.13456  
5. **Nichol, A., & Dhariwal, P.** (2021). *Improved Denoising Diffusion Probabilistic Models*. ICML.  
   https://arxiv.org/abs/2102.09672  
6. **Dhariwal, P., & Nichol, A.** (2021). *Diffusion Models Beat GANs on Image Synthesis*. NeurIPS.  
   https://arxiv.org/abs/2105.05233  

### Latent diffusion & Stable Diffusion
7. **Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B.** (2022). *High-Resolution Image Synthesis with Latent Diffusion Models* (LDM / basis of Stable Diffusion). CVPR.  
   https://arxiv.org/abs/2112.10752  
8. **Esser, P., Rombach, R., & Ommer, B.** (2021). *Taming Transformers for High-Resolution Image Synthesis* (VQGAN — perceptual compression used with LDMs). CVPR.  
   https://arxiv.org/abs/2012.09841  
9. **Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., & Chen, M.** (2022). *Hierarchical Text-Conditional Image Generation with CLIP Latents* (unCLIP / DALL·E 2).  
   https://arxiv.org/abs/2204.06125  
10. **Saharia, C., et al.** (2022). *Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding* (Imagen). NeurIPS.  
    https://arxiv.org/abs/2205.11487  
11. **Podell, D., et al.** (2023). *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis*.  
    https://arxiv.org/abs/2307.01952  
12. **Stability AI / CompVis** — Stable Diffusion model cards & code (open LDM release):  
    - https://github.com/CompVis/stable-diffusion  
    - https://github.com/Stability-AI/stablediffusion  
13. **Ho, J., & Salimans, T.** (2022). *Classifier-Free Diffusion Guidance* (CFG used heavily in Stable Diffusion).  
    https://arxiv.org/abs/2207.12598  
14. **Radford, A., et al.** (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP — text conditioning). ICML.  
    https://arxiv.org/abs/2103.00020  

### Flow matching (incl. modern latent generators)
15. **Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M.** (2023). *Flow Matching for Generative Modeling*. ICLR.  
    https://arxiv.org/abs/2210.02747  
16. **Liu, X., Gong, C., & Liu, Q.** (2022). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*.  
    https://arxiv.org/abs/2209.03003  
17. **Esser, P., et al.** (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (SD3).  
    https://arxiv.org/abs/2403.03206  

### Conditioning & guidance (Stable Diffusion practice)
18. **Zhang, L., Rao, A., & Agrawala, M.** (2023). *Adding Conditional Control to Text-to-Image Diffusion Models* (ControlNet). ICCV.  
    https://arxiv.org/abs/2302.05543  
19. **Ruiz, N., et al.** (2023). *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation*. CVPR.  
    https://arxiv.org/abs/2208.12242  
20. **Hu, E. J., et al.** (2022). *LoRA: Low-Rank Adaptation of Large Language Models* (widely used to fine-tune SD).  
    https://arxiv.org/abs/2106.09685  

### Interactive / educational tools
21. **Helbling, A., & Chau, D. H.** (2025). *Diffusion Explorer: Interactive Exploration of Diffusion Models*.  
    https://arxiv.org/abs/2507.01178  
22. **Lee, S., et al.** *Diffusion Explainer* (Stable Diffusion pipeline visualization).  
    https://poloclub.github.io/diffusion-explainer  
23. **Diffusion Meets Flow Matching** (equivalence notes).  
    https://diffusionflow.github.io/  

---

## BibTeX (Stable Diffusion / LDM–centric)

```bibtex
@inproceedings{rombach2022ldm,
  title     = {High-Resolution Image Synthesis with Latent Diffusion Models},
  author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik
               and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle = {CVPR},
  year      = {2022},
  url       = {https://arxiv.org/abs/2112.10752}
}

@inproceedings{esser2021taming,
  title     = {Taming Transformers for High-Resolution Image Synthesis},
  author    = {Esser, Patrick and Rombach, Robin and Ommer, Bj{\"o}rn},
  booktitle = {CVPR},
  year      = {2021},
  url       = {https://arxiv.org/abs/2012.09841}
}

@article{podell2023sdxl,
  title  = {SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis},
  author = {Podell, Dustin and English, Zion and Lacey, Kyle and Blattmann, Andreas
            and Dockhorn, Tim and M{\"u}ller, Jonas and Penna, Joe and Rombach, Robin},
  journal = {arXiv preprint arXiv:2307.01952},
  year   = {2023}
}

@article{esser2024sd3,
  title  = {Scaling Rectified Flow Transformers for High-Resolution Image Synthesis},
  author = {Esser, Patrick and Kulal, Sumith and Blattmann, Andreas and others},
  journal = {arXiv preprint arXiv:2403.03206},
  year   = {2024}
}

@article{ho2022cfg,
  title  = {Classifier-Free Diffusion Guidance},
  author = {Ho, Jonathan and Salimans, Tim},
  journal = {arXiv preprint arXiv:2207.12598},
  year   = {2022}
}

@inproceedings{ho2020ddpm,
  title     = {Denoising Diffusion Probabilistic Models},
  author    = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle = {NeurIPS},
  year      = {2020}
}

@inproceedings{song2021score,
  title     = {Score-Based Generative Modeling through Stochastic Differential Equations},
  author    = {Song, Yang and Sohl-Dickstein, Jascha and Kingma, Diederik P.
               and Kumar, Abhishek and Ermon, Stefano and Poole, Ben},
  booktitle = {ICLR},
  year      = {2021}
}

@inproceedings{lipman2023flow,
  title     = {Flow Matching for Generative Modeling},
  author    = {Lipman, Yaron and Chen, Ricky T. Q. and Ben-Hamu, Heli
               and Nickel, Maximilian and Le, Matt},
  booktitle = {ICLR},
  year      = {2023}
}
```
