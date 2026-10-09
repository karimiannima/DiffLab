# Flow matching and score diffusion demo

NumPy-only walkthrough of the two regression targets used in DiffLab.

Score diffusion fits the noise in a variance-preserving kernel, then samples with the DDIM step (the closed form of the probability-flow ODE). Flow matching fits the straight-path velocity \(x_1 - x_0\) and samples with Euler.

```bash
python demo_flow_score.py
```

Needs NumPy and Matplotlib. Regenerates:

- `demo_sde_trajectories.png` — forward SDE, reverse SDE, probability-flow ODE
- `demo_flow_ribbon.png` — straight conditional paths vs the curved marginal flow
- `demo_samples.png` — 2D three-blob samples from both models
- `demo_flow_path.png` — flow-matching snapshots from noise to data

`flow_matching_score_diffusion.pptx` is the lecture deck for the same derivations (Anderson reversal, denoising score matching, continuity equation, conditional flow matching).
