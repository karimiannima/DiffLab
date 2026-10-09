"""
2D demo: score diffusion and conditional flow matching.

Both models regress a vector field on the same idea of a probability path.
Only the label changes.

  Score diffusion (VP kernel)
    x_t = alpha_t * x_data + sigma_t * eps
    target = eps                         # epsilon parameterization of the score
    sample with the DDIM / probability-flow step

  Flow matching (straight interpolant)
    x_t = (1 - t) * x_noise + t * x_data
    target = x_data - x_noise
    sample with dx/dt = v_theta

Also writes the textbook ribbon used by Song et al. and
https://www.physicsbaseddeeplearning.org/probmodels-score.html
time on x, state on y, p_t as the colored band, SDE vs probability-flow ODE.

Run:
  python demo_flow_score.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(7)


def sample_data(n: int, rng: np.random.Generator = RNG) -> np.ndarray:
    centers = np.array([[-2.2, -0.6], [0.2, 2.0], [2.2, -0.8]], dtype=np.float32)
    idx = rng.integers(0, 3, size=n)
    return centers[idx] + 0.28 * rng.normal(size=(n, 2)).astype(np.float32)


def sample_noise(n: int, rng: np.random.Generator = RNG) -> np.ndarray:
    return rng.normal(size=(n, 2)).astype(np.float32)


def timestep_embed(t: np.ndarray, n_freq: int = 8) -> np.ndarray:
    freqs = (2.0 ** np.arange(n_freq)).astype(np.float32) * math.pi
    ang = t * freqs[None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)


class Field:
    """Two-layer MLP. Input is [x, time embedding]."""

    def __init__(self, hidden: int = 128, n_freq: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        d_in = 2 + 2 * n_freq
        self.n_freq = n_freq
        self.w1 = rng.normal(0, math.sqrt(2 / d_in), size=(d_in, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.w2 = rng.normal(0, math.sqrt(2 / hidden), size=(hidden, hidden)).astype(np.float32)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.w3 = rng.normal(0, math.sqrt(2 / hidden), size=(hidden, 2)).astype(np.float32)
        self.b3 = np.zeros(2, dtype=np.float32)

    def forward(self, x: np.ndarray, t: np.ndarray):
        h_in = np.concatenate([x, timestep_embed(t, self.n_freq)], axis=1)
        z1 = h_in @ self.w1 + self.b1
        h1 = np.maximum(z1, 0)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(z2, 0)
        out = h2 @ self.w3 + self.b3
        return out, (h_in, z1, h1, z2, h2)

    def backward(self, grad_out: np.ndarray, cache, lr: float):
        h_in, z1, h1, z2, h2 = cache
        n = grad_out.shape[0]
        gw3 = h2.T @ grad_out / n
        gb3 = grad_out.mean(axis=0)
        gh2 = grad_out @ self.w3.T
        gz2 = gh2 * (z2 > 0)
        gw2 = h1.T @ gz2 / n
        gb2 = gz2.mean(axis=0)
        gh1 = gz2 @ self.w2.T
        gz1 = gh1 * (z1 > 0)
        gw1 = h_in.T @ gz1 / n
        gb1 = gz1.mean(axis=0)
        self.w3 -= lr * gw3
        self.b3 -= lr * gb3
        self.w2 -= lr * gw2
        self.b2 -= lr * gb2
        self.w1 -= lr * gw1
        self.b1 -= lr * gb1

    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        out, _ = self.forward(x, t)
        return out


def vp_alpha_sigma(t: np.ndarray, beta_min: float = 0.1, beta_max: float = 20.0):
    log_alpha = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
    alpha = np.exp(log_alpha)
    sigma = np.sqrt(np.maximum(1.0 - alpha**2, 1e-5))
    return alpha.astype(np.float32), sigma.astype(np.float32)


def train_score(steps: int = 3000, batch: int = 512, lr: float = 1.5e-2) -> Field:
    """Denoising score matching, epsilon parameterization.

    Conditional score of the VP kernel is -eps / sigma. Predicting eps is the
    same regression with a unit-scale target.
    """
    model = Field(seed=1)
    rng = np.random.default_rng(1)
    for step in range(steps):
        x0 = sample_data(batch, rng)
        eps = sample_noise(batch, rng)
        t = rng.uniform(0.02, 1.0, size=(batch, 1)).astype(np.float32)
        alpha, sigma = vp_alpha_sigma(t)
        x_t = alpha * x0 + sigma * eps
        pred, cache = model.forward(x_t, t)
        model.backward(2.0 * (pred - eps), cache, lr)
        if step % 500 == 0 or step == steps - 1:
            print(f"  score  step {step:4d}  eps-mse {float(np.mean((pred - eps) ** 2)):.4f}")
    return model


def train_flow(steps: int = 2500, batch: int = 512, lr: float = 2e-2) -> Field:
    """Conditional flow matching, independent straight coupling."""
    model = Field(seed=2)
    rng = np.random.default_rng(2)
    for step in range(steps):
        x1 = sample_data(batch, rng)
        x0 = sample_noise(batch, rng)
        t = rng.uniform(0.0, 1.0, size=(batch, 1)).astype(np.float32)
        x_t = (1.0 - t) * x0 + t * x1
        target = x1 - x0
        pred, cache = model.forward(x_t, t)
        model.backward(2.0 * (pred - target), cache, lr)
        if step % 500 == 0 or step == steps - 1:
            print(f"  flow   step {step:4d}  mse {float(np.mean((pred - target) ** 2)):.4f}")
    return model


def sample_score_pf(model: Field, n: int = 1500, n_steps: int = 40) -> np.ndarray:
    """DDIM step: the closed form of the probability-flow ODE for the VP kernel."""
    x = sample_noise(n)
    ts = np.linspace(1.0, 0.02, n_steps + 1).astype(np.float32)
    for i in range(n_steps):
        t = np.full((n, 1), ts[i], dtype=np.float32)
        t_next = np.full((n, 1), ts[i + 1], dtype=np.float32)
        alpha, sigma = vp_alpha_sigma(t)
        alpha_next, sigma_next = vp_alpha_sigma(t_next)
        eps = model.predict(x, t)
        x0 = np.clip((x - sigma * eps) / np.maximum(alpha, 1e-3), -6, 6)
        x = alpha_next * x0 + sigma_next * eps
    return x


def sample_flow(model: Field, n: int = 1500, n_steps: int = 40) -> np.ndarray:
    x = sample_noise(n)
    ts = np.linspace(0.0, 1.0, n_steps + 1).astype(np.float32)
    for i in range(n_steps):
        t = np.full((n, 1), ts[i], dtype=np.float32)
        x = x + model.predict(x, t) * (ts[i + 1] - ts[i])
    return x


def flow_snapshots(model: Field, n: int = 800, n_steps: int = 8) -> list[np.ndarray]:
    x = sample_noise(n)
    frames = [x.copy()]
    ts = np.linspace(0.0, 1.0, n_steps + 1).astype(np.float32)
    for i in range(n_steps):
        t = np.full((n, 1), ts[i], dtype=np.float32)
        x = x + model.predict(x, t) * (ts[i + 1] - ts[i])
        frames.append(x.copy())
    return frames


def _panel(ax, pts, title, color):
    ax.scatter(pts[:, 0], pts[:, 1], s=6, c=color, alpha=0.55, linewidths=0)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-3.6, 3.8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Textbook ribbon. Time on x, 1D state on y, p_t as the colored band.
# Layout of Song et al., as used in the PBDL score chapter.
# ---------------------------------------------------------------------------

def _beta(t, beta_min=0.1, beta_max=8.0):
    return beta_min + float(t) * (beta_max - beta_min)


def _alpha_sigma_scalar(t, beta_min=0.1, beta_max=8.0):
    integ = beta_min * t + 0.5 * (beta_max - beta_min) * t**2
    alpha = np.exp(-0.5 * integ)
    return alpha, np.sqrt(np.maximum(1.0 - np.exp(-integ), 1e-8))


def mixture_density_and_score(x, t, mus, stds, weights, beta_min=0.1, beta_max=8.0):
    alpha, sigma = _alpha_sigma_scalar(t, beta_min, beta_max)
    comps, score_num = [], []
    for mu, std, w in zip(mus, stds, weights):
        mean = alpha * mu
        var = alpha**2 * std**2 + sigma**2
        phi = np.exp(-0.5 * np.log(2 * math.pi * var) - 0.5 * (x - mean) ** 2 / var)
        comps.append(w * phi)
        score_num.append(w * phi * (-(x - mean) / var))
    p = np.sum(comps, axis=0)
    return p, np.sum(score_num, axis=0) / np.maximum(p, 1e-12)


def song_style_figure(path: Path):
    """Forward SDE, reverse SDE, probability-flow ODE. Exact VP mixture score."""
    beta_min, beta_max = 0.1, 8.0
    mus = np.array([-1.55, 1.45])
    stds = np.array([0.22, 0.26])
    weights = np.array([0.5, 0.5])
    rng = np.random.default_rng(4)

    n_t, n_x = 280, 240
    t_grid = np.linspace(0.0, 1.0, n_t)
    x_grid = np.linspace(-3.2, 3.2, n_x)
    tt, xx = np.meshgrid(t_grid, x_grid)
    density, _ = mixture_density_and_score(xx, tt, mus, stds, weights, beta_min, beta_max)
    log_d = np.log(density + 1e-8)

    def sample_x0(n):
        idx = rng.choice(len(mus), size=n, p=weights)
        return rng.normal(mus[idx], stds[idx])

    n_steps = 220

    def euler_forward(x):
        ts = np.linspace(0.0, 1.0, n_steps + 1)
        xs = [x.copy()]
        for i in range(n_steps):
            dt = ts[i + 1] - ts[i]
            b = _beta(ts[i], beta_min, beta_max)
            x = x - 0.5 * b * x * dt + np.sqrt(b * dt) * rng.normal(size=x.shape)
            xs.append(x.copy())
        return ts, np.stack(xs, axis=0)

    def euler_back(x, stochastic):
        ts = np.linspace(1.0, 0.0, n_steps + 1)
        xs = [x.copy()]
        for i in range(n_steps):
            t = max(ts[i], 1e-3)
            dt = ts[i] - ts[i + 1]
            b = _beta(t, beta_min, beta_max)
            _, score = mixture_density_and_score(x, t, mus, stds, weights, beta_min, beta_max)
            coef = 1.0 if stochastic else 0.5
            x = x + (0.5 * b * x + coef * b * score) * dt
            if stochastic:
                x = x + np.sqrt(b * dt) * rng.normal(size=x.shape)
            xs.append(x.copy())
        return ts[::-1], np.stack(xs, axis=0)[::-1]

    def euler_pf_forward(x):
        ts = np.linspace(0.0, 1.0, n_steps + 1)
        xs = [x.copy()]
        for i in range(n_steps):
            dt = ts[i + 1] - ts[i]
            b = _beta(ts[i], beta_min, beta_max)
            _, score = mixture_density_and_score(
                x, max(ts[i], 1e-3), mus, stds, weights, beta_min, beta_max
            )
            x = x + (-0.5 * b * x - 0.5 * b * score) * dt
            xs.append(x.copy())
        return ts, np.stack(xs, axis=0)

    n_sde, n_ode = 12, 6
    t_fwd, paths_fwd = euler_forward(sample_x0(n_sde))
    _, pf_fwd = euler_pf_forward(sample_x0(n_ode))
    t_rev, paths_sde = euler_back(rng.normal(size=n_sde), stochastic=True)
    _, paths_pf = euler_back(rng.normal(size=n_ode), stochastic=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), dpi=160, sharey=True)
    extent = [0.0, 1.0, x_grid[0], x_grid[-1]]
    vmin, vmax = np.percentile(log_d, 12), np.percentile(log_d, 99.6)
    specs = [
        (axes[0], "Forward SDE", r"$dx = f(x,t)\,dt + g(t)\,dw$"),
        (axes[1], "Reverse SDE", r"$dx = [f - g(t)^2\nabla_x\log p_t(x)]\,dt + g(t)\,d\bar{w}$"),
    ]
    for ax, title, eq in specs:
        ax.imshow(
            log_d, origin="lower", extent=extent, aspect="auto",
            cmap="turbo", vmin=vmin, vmax=vmax, interpolation="bilinear",
        )
        ax.set_title(
            title, fontsize=14, color="white", pad=18,
            bbox=dict(facecolor="black", edgecolor="none", pad=2.5),
        )
        ax.text(0.5, 1.045, eq, transform=ax.transAxes, ha="center", va="bottom", fontsize=10)

    for i in range(n_sde):
        axes[0].plot(t_fwd, paths_fwd[:, i], color="#7a1020", lw=1.05, alpha=0.92)
        axes[1].plot(t_rev, paths_sde[:, i], color="#7a1020", lw=1.05, alpha=0.92)
    for i in range(n_ode):
        axes[0].plot(t_fwd, pf_fwd[:, i], color="#111111", lw=1.7)
        axes[1].plot(t_rev, paths_pf[:, i], color="#111111", lw=1.7)

    axes[1].plot([], [], color="#7a1020", lw=1.6, label="SDE")
    axes[1].plot([], [], color="#111111", lw=1.8, label="Probability flow ODE")
    leg = axes[1].legend(loc="upper right", frameon=True, fontsize=8, facecolor="white")
    leg.get_frame().set_edgecolor("#dddddd")

    sides = [
        (axes[0], "Data\n$x(0)$", "Prior\n$x(T)$", r"$p_0(x)\ \rightarrow\ p_t(x)\ \rightarrow\ p_T(x)$"),
        (axes[1], "Prior\n$x(T)$", "Data\n$x(0)$", r"$p_T(x)\ \rightarrow\ p_t(x)\ \rightarrow\ p_0(x)$"),
    ]
    for ax, left, right, xlab in sides:
        ax.set_xlim(0, 1)
        ax.set_ylim(x_grid[0], x_grid[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(-0.03, 0.5, left, transform=ax.transAxes, ha="right", va="center", fontsize=9)
        ax.text(1.03, 0.5, right, transform=ax.transAxes, ha="left", va="center", fontsize=9)
        ax.set_xlabel(xlab, fontsize=10, labelpad=6)

    fig.suptitle(
        "Same marginals: the SDE wiggles, the probability-flow ODE does not",
        fontsize=13, y=1.06,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"wrote {path}")


def flow_ribbon_figure(path: Path):
    """Same ribbon for flow matching: straight conditionals, curved marginal."""
    mus = np.array([-1.55, 1.45])
    stds = np.array([0.22, 0.26])
    weights = np.array([0.5, 0.5])
    rng = np.random.default_rng(5)
    n_t, n_x = 240, 220
    t_grid = np.linspace(0.0, 1.0, n_t)
    x_grid = np.linspace(-3.2, 3.2, n_x)
    tt, xx = np.meshgrid(t_grid, x_grid)
    comps = []
    for mu, std, w in zip(mus, stds, weights):
        mean = tt * mu
        var = (1.0 - tt) ** 2 + (tt * std) ** 2
        phi = np.exp(-0.5 * np.log(2 * math.pi * var) - 0.5 * (xx - mean) ** 2 / var)
        comps.append(w * phi)
    log_d = np.log(np.sum(comps, axis=0) + 1e-8)

    def sample_x1(n):
        idx = rng.choice(len(mus), size=n, p=weights)
        return rng.normal(mus[idx], stds[idx])

    n_paths = 10
    x0 = rng.normal(size=n_paths)
    x1 = sample_x1(n_paths)
    ts = np.linspace(0.0, 1.0, 90)
    cond = (1.0 - ts[:, None]) * x0[None, :] + ts[:, None] * x1[None, :]

    def marginal_velocity(x, t):
        comps_i, num = [], []
        for mu, std, w in zip(mus, stds, weights):
            mean = t * mu
            var = (1.0 - t) ** 2 + (t * std) ** 2
            phi = np.exp(-0.5 * np.log(2 * math.pi * var) - 0.5 * (x - mean) ** 2 / var)
            post_mean = (t * std**2 * x + (1.0 - t) ** 2 * mu) / max(var, 1e-8)
            u = (post_mean - x) / max(1.0 - t, 0.04)
            comps_i.append(w * phi)
            num.append(w * phi * u)
        return np.sum(num, axis=0) / np.maximum(np.sum(comps_i, axis=0), 1e-12)

    x = rng.normal(size=n_paths)
    marg = [x.copy()]
    for i in range(len(ts) - 1):
        x = x + marginal_velocity(x, ts[i]) * (ts[i + 1] - ts[i])
        marg.append(x.copy())
    marg = np.stack(marg, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), dpi=160, sharey=True)
    extent = [0.0, 1.0, x_grid[0], x_grid[-1]]
    vmin, vmax = np.percentile(log_d, 12), np.percentile(log_d, 99.6)
    for ax, title, eq in (
        (axes[0], "Conditional paths", r"$x_t=(1-t)\,x_0+t\,x_1$"),
        (axes[1], "Marginal flow", r"$dx/dt=\mathbb{E}[x_1-x_0\mid x_t=x]$"),
    ):
        ax.imshow(
            log_d, origin="lower", extent=extent, aspect="auto",
            cmap="turbo", vmin=vmin, vmax=vmax, interpolation="bilinear",
        )
        ax.set_title(
            title, fontsize=14, color="white", pad=18,
            bbox=dict(facecolor="black", edgecolor="none", pad=2.5),
        )
        ax.text(0.5, 1.045, eq, transform=ax.transAxes, ha="center", va="bottom", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(x_grid[0], x_grid[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(-0.03, 0.5, "Noise\n$x_0$", transform=ax.transAxes, ha="right", va="center", fontsize=9)
        ax.text(1.03, 0.5, "Data\n$x_1$", transform=ax.transAxes, ha="left", va="center", fontsize=9)
    for i in range(n_paths):
        axes[0].plot(ts, cond[:, i], color="#7a1020", lw=1.15, alpha=0.92)
        axes[1].plot(ts, marg[:, i], color="#111111", lw=1.7)
    axes[0].set_xlabel("straight given the pair, crossing in the mixture", fontsize=10)
    axes[1].set_xlabel("the field conditional flow matching actually learns", fontsize=10)
    fig.suptitle(
        "Flow matching, same picture: conditionals are straight, the marginal is not",
        fontsize=13, y=1.06,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"wrote {path}")


def main():
    song_style_figure(OUT / "demo_sde_trajectories.png")
    flow_ribbon_figure(OUT / "demo_flow_ribbon.png")

    print("training score model")
    score = train_score()
    print("training flow model")
    flow = train_flow()

    data = sample_data(1500)
    print("sampling")
    score_samples = sample_score_pf(score)
    flow_samples = sample_flow(flow)
    frames = flow_snapshots(flow)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), dpi=140)
    _panel(axes[0], data, "Data  p_1", "#16263D")
    _panel(axes[1], score_samples, "Score diffusion  (PF-ODE)", "#E8624D")
    _panel(axes[2], flow_samples, "Flow matching  (Euler)", "#2F6F4E")
    fig.suptitle("Same three-blob target, two regression labels", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "demo_samples.png", bbox_inches="tight", facecolor="white")
    plt.close()

    fig, axes = plt.subplots(1, 5, figsize=(12.5, 2.8), dpi=140)
    show = [0, 2, 4, 6, 8]
    for ax, k in zip(axes, show):
        _panel(ax, frames[k], f"t = {k / 8:.2f}", "#1F4E79")
    fig.suptitle("Conditional flow matching: noise transported to data", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "demo_flow_path.png", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"wrote {OUT / 'demo_samples.png'}")
    print(f"wrote {OUT / 'demo_flow_path.png'}")


if __name__ == "__main__":
    main()
