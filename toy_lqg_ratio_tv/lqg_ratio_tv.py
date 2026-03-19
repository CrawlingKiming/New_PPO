#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _get_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@dataclass
class LQG1D:
    a: float = 0.95
    b: float = 0.5
    q: float = 1.0
    r: float = 0.1
    sigma_xi: float = 0.1
    gamma: float = 0.99

    def step(self, s: float, u: float, rng: np.random.Generator) -> tuple[float, float]:
        xi = rng.normal(0.0, self.sigma_xi)
        s_next = self.a * s + self.b * u + xi
        reward = -(self.q * s * s + self.r * u * u)
        return float(s_next), float(reward)


class LinearGaussianPolicy:
    def __init__(self, k: float, sigma: float):
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        self.k = float(k)
        self.sigma = float(sigma)

    def sample(self, s: float, rng: np.random.Generator) -> float:
        mean = -self.k * s
        return float(rng.normal(mean, self.sigma))

    def logpdf(self, u: np.ndarray | float, s: np.ndarray | float) -> np.ndarray | float:
        mean = -self.k * np.asarray(s)
        z = (np.asarray(u) - mean) / self.sigma
        return -0.5 * np.log(2.0 * np.pi) - np.log(self.sigma) - 0.5 * z * z


def lqr_gain_1d(
    a: float,
    b: float,
    q: float,
    r: float,
    gamma: float,
    iters: int = 5000,
    tol: float = 1e-12,
) -> tuple[float, float]:
    P = float(q)
    for _ in range(iters):
        denom = r + gamma * (b * b) * P
        P_new = q + gamma * (a * a) * P - (gamma * a * b * P) ** 2 / denom
        if abs(P_new - P) < tol:
            P = P_new
            break
        P = P_new
    k_star = (gamma * a * b * P) / (r + gamma * (b * b) * P)
    return float(k_star), float(P)


def collect_behavior_samples(
    env: LQG1D,
    mu: LinearGaussianPolicy,
    T: int,
    burnin: int,
    s0: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = float(s0)
    states: list[float] = []
    actions: list[float] = []

    for t in range(T + burnin):
        u = mu.sample(s, rng)
        s_next, _ = env.step(s, u, rng)
        if t >= burnin:
            states.append(s)
            actions.append(u)
        s = s_next

    return np.asarray(states, dtype=np.float64), np.asarray(actions, dtype=np.float64)


def ratios_from_samples(
    pi: LinearGaussianPolicy,
    mu: LinearGaussianPolicy,
    states: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    return np.exp(pi.logpdf(actions, states) - mu.logpdf(actions, states))


def estimate_E_abs_R_minus_1(ratios: np.ndarray) -> float:
    return float(np.mean(np.abs(ratios - 1.0)))


def estimate_tv_from_ratios(ratios: np.ndarray) -> float:
    return 0.5 * estimate_E_abs_R_minus_1(ratios)


def parse_float_list(raw: str | None) -> list[float]:
    if raw is None:
        return []
    vals: list[float] = []
    for piece in raw.split(","):
        p = piece.strip()
        if not p:
            continue
        vals.append(float(p))
    return vals


def build_sparse_k_grid(
    mu_k: float,
    k_star: float,
    k_points: int,
    k_min: float | None = None,
    k_max: float | None = None,
) -> np.ndarray:
    if k_points < 3:
        raise ValueError("k_points must be >= 3 for a meaningful sparse sweep")

    if k_min is None or k_max is None:
        span = max(2.0, 2.0 * abs(k_star - mu_k))
        auto_min = mu_k - 0.5 * span
        auto_max = mu_k + span
        k_min = auto_min if k_min is None else k_min
        k_max = auto_max if k_max is None else k_max

    if k_max <= k_min:
        raise ValueError("k_max must be greater than k_min")

    base = np.linspace(float(k_min), float(k_max), int(k_points))
    return np.unique(np.concatenate([base, np.asarray([mu_k, k_star], dtype=np.float64)]))


def value_coefficients_linear_policy(env: LQG1D, policy: LinearGaussianPolicy) -> tuple[float, float]:
    acl = env.a - env.b * policy.k
    contraction = env.gamma * acl * acl
    if contraction >= 1.0:
        raise ValueError(
            "Policy is not stable enough for finite discounted quadratic value: "
            f"gamma*(a-b*k)^2={contraction:.6f} >= 1"
        )

    P = (env.q + env.r * policy.k * policy.k) / (1.0 - contraction)
    noise_var = (env.b * policy.sigma) ** 2 + env.sigma_xi**2
    C = (-env.r * policy.sigma**2 - env.gamma * P * noise_var) / (1.0 - env.gamma)
    return float(P), float(C)


def advantage_under_linear_policy(
    env: LQG1D,
    P: float,
    C: float,
    states: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    states = np.asarray(states)
    actions = np.asarray(actions)
    v_s = -P * states * states + C
    s_next_mean = env.a * states + env.b * actions
    q_sa = (
        -(env.q * states * states + env.r * actions * actions)
        + env.gamma * (-P * (s_next_mean * s_next_mean + env.sigma_xi**2) + C)
    )
    return q_sa - v_s


def optimal_advantage(env: LQG1D, P_star: float, s: float | np.ndarray, u: np.ndarray) -> np.ndarray:
    s = np.asarray(s)
    u = np.asarray(u)
    denom = env.r + env.gamma * (env.b * env.b) * P_star
    k_star = (env.gamma * env.a * env.b * P_star) / denom
    return -denom * (u + k_star * s) ** 2


def build_tv_constraint_values(
    max_tv: float,
    tv_constraints_raw: str | None,
    tv_constraint_single: float | None,
    epsilon: float | None,
    tv_constraint_points: int,
) -> np.ndarray:
    vals = parse_float_list(tv_constraints_raw)
    if tv_constraint_single is not None:
        vals.append(float(tv_constraint_single))
    if epsilon is not None:
        vals.append(0.5 * float(epsilon))

    if not vals:
        points = max(2, int(tv_constraint_points))
        vals = list(np.linspace(0.0, max_tv, points))

    arr = np.asarray(vals, dtype=np.float64)
    arr = np.clip(arr, 0.0, max_tv)
    arr = np.unique(arr)

    if arr.size == 0:
        arr = np.asarray([0.0], dtype=np.float64)
    if arr[0] > 0.0:
        arr = np.concatenate([np.asarray([0.0], dtype=np.float64), arr])
    if arr[-1] < max_tv:
        arr = np.concatenate([arr, np.asarray([max_tv], dtype=np.float64)])
    return arr


def _density_curve(
    ratios: np.ndarray,
    x_max: float,
    bins: int = 180,
    smooth_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    ratios = np.asarray(ratios)
    ratios = ratios[np.isfinite(ratios)]
    ratios = ratios[ratios >= 0.0]
    if ratios.size == 0:
        x = np.linspace(0.0, max(1.0, x_max), bins)
        return x, np.zeros_like(x)

    hi = max(float(x_max), 1.05)
    edges = np.linspace(0.0, hi, bins + 1)
    hist = np.histogram(np.clip(ratios, 0.0, hi), bins=edges, density=True)[0]
    centers = 0.5 * (edges[:-1] + edges[1:])

    w = max(1, int(smooth_window))
    if w > 1:
        kernel = np.ones(w, dtype=np.float64) / float(w)
        hist = np.convolve(hist, kernel, mode="same")
    return centers, hist


def plot_ratio_hist_overlay(
    tv_values: np.ndarray,
    ratio_arrays: list[np.ndarray],
    out_path: Path,
    max_curves: int = 6,
) -> None:
    plt = _get_plt()
    if not ratio_arrays:
        return

    idx = np.linspace(0, len(ratio_arrays) - 1, min(max_curves, len(ratio_arrays)), dtype=int)
    hi = max(float(np.quantile(np.asarray(ratio_arrays[i]), 0.995)) for i in idx)
    hi = max(hi, 1.05)

    plt.figure(figsize=(8, 5))
    for i in idx:
        x, y = _density_curve(np.asarray(ratio_arrays[i]), x_max=hi, bins=180, smooth_window=7)
        plt.plot(x, y, linewidth=1.6, label=f"TV~{tv_values[i]:.3f}")
    plt.axvline(1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.9)
    plt.xlim(0.0, hi)
    plt.xlabel("Ratio R = pi(u|s) / mu(u|s)")
    plt.ylabel("Density")
    plt.title("Ratio Density Overlay")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_ratio_quantiles_vs_tv(tv_values: np.ndarray, ratio_arrays: list[np.ndarray], out_path: Path) -> None:
    plt = _get_plt()
    if not ratio_arrays:
        return

    q10, q50, q90 = [], [], []
    for ratios in ratio_arrays:
        lr = np.log10(np.clip(np.asarray(ratios), 1e-12, None))
        q10.append(float(np.quantile(lr, 0.10)))
        q50.append(float(np.quantile(lr, 0.50)))
        q90.append(float(np.quantile(lr, 0.90)))

    plt.figure(figsize=(8, 5))
    plt.plot(tv_values, q10, marker="o", label="10%")
    plt.plot(tv_values, q50, marker="o", label="50%")
    plt.plot(tv_values, q90, marker="o", label="90%")
    plt.xlabel("Empirical TV(pi, mu)")
    plt.ylabel("log10(R) quantiles")
    plt.title("Ratio Quantiles vs TV")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_tv_identity(tv_values: np.ndarray, e_abs_values: np.ndarray, out_path: Path) -> None:
    plt = _get_plt()
    if tv_values.size == 0:
        return
    x_line = np.linspace(0.0, float(max(np.max(tv_values), 1e-8)), 200)

    plt.figure(figsize=(8, 5))
    plt.plot(tv_values, e_abs_values, marker="o", linewidth=1.8, label="Empirical E[|R-1|]")
    plt.plot(x_line, 2.0 * x_line, linestyle="--", linewidth=1.4, label="2*TV")
    plt.xlabel("Empirical TV(pi, mu)")
    plt.ylabel("Value")
    plt.title("Identity Check: E[|R-1|] = 2 * TV")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_advantage_curve(env: LQG1D, P_star: float, s_fixed: float, out_path: Path) -> None:
    plt = _get_plt()
    u_grid = np.linspace(-3.0, 3.0, 500)
    adv = optimal_advantage(env, P_star, s_fixed, u_grid)
    u_opt = -(env.gamma * env.a * env.b * P_star) / (env.r + env.gamma * (env.b * env.b) * P_star) * s_fixed

    plt.figure(figsize=(8, 5))
    plt.plot(u_grid, adv, linewidth=2.0)
    plt.axvline(u_opt, linestyle="--", linewidth=1.3, label=f"u*={u_opt:.3f}")
    plt.xlabel("Action u")
    plt.ylabel("A*(s, u)")
    plt.title(f"Optimal Advantage at s={s_fixed:.2f}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_value_vs_tv(
    tv_values: np.ndarray,
    values: np.ndarray,
    ks: np.ndarray,
    tv_ref: float,
    out_path: Path,
) -> None:
    plt = _get_plt()
    if tv_values.size == 0:
        return

    plt.figure(figsize=(8, 5))
    sc = plt.scatter(tv_values, values, c=ks, cmap="viridis", s=40, alpha=0.9)
    plt.axvline(tv_ref, color="tab:red", linestyle="--", linewidth=1.4, label="TV reference")
    cbar = plt.colorbar(sc)
    cbar.set_label("Policy gain k")
    plt.xlabel("Empirical TV(pi_k, mu)")
    plt.ylabel("Analytic V_pi_k(s0)")
    plt.title("Sparse k Sweep: Value vs TV")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_selected_k_vs_tv_constraint(
    constraints: np.ndarray,
    selected_ks: np.ndarray,
    selected_tvs: np.ndarray,
    out_path: Path,
) -> None:
    plt = _get_plt()
    if constraints.size == 0:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(constraints, selected_ks, marker="o", label="Selected k")
    plt.plot(constraints, selected_tvs, marker="s", label="Realized TV")
    plt.xlabel("TV constraint")
    plt.ylabel("k / TV")
    plt.title("Selected Policy vs TV Constraint")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_ratio_hist(
    ratios: np.ndarray,
    out_path: Path,
    title: str,
    histtype: str,
    alpha: float,
    color: str,
    x_max: float,
) -> None:
    plt = _get_plt()
    ratios = np.asarray(ratios)
    ratios = ratios[np.isfinite(ratios)]
    ratios = ratios[ratios >= 0.0]

    plt.figure(figsize=(8, 5))
    if ratios.size == 0:
        plt.text(0.5, 0.5, "No samples in this split.", ha="center", va="center")
        plt.title(title)
        plt.xlabel("Ratio R = pi(u|s) / mu(u|s)")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    hi = max(float(x_max), 1.05)
    bins = np.linspace(0.0, hi, 121)
    plt.hist(
        np.clip(ratios, 0.0, hi),
        bins=bins,
        density=True,
        histtype=histtype,
        linewidth=1.5 if histtype == "step" else 1.0,
        color=color,
        alpha=alpha,
    )
    plt.axvline(1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.9)
    plt.xlim(0.0, hi)
    plt.xlabel("Ratio R = pi(u|s) / mu(u|s)")
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_truncation_overlay_at_one(
    raw_ratios: np.ndarray,
    trunc_ratios: np.ndarray,
    out_path: Path,
    title: str,
    raw_color: str,
) -> None:
    plt = _get_plt()
    raw = np.asarray(raw_ratios)
    trunc = np.asarray(trunc_ratios)
    raw = raw[np.isfinite(raw)]
    trunc = trunc[np.isfinite(trunc)]
    raw = raw[raw >= 0.0]
    trunc = trunc[trunc >= 0.0]

    plt.figure(figsize=(8, 5))
    if raw.size == 0:
        plt.text(0.5, 0.5, "No samples in this split.", ha="center", va="center")
        plt.title(title)
        plt.xlabel("Ratio R = pi(u|s) / mu(u|s)")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    hi = max(float(np.quantile(raw, 0.995)), 1.05)
    x_raw, y_raw = _density_curve(raw, x_max=hi, bins=180, smooth_window=7)
    x_tr, y_tr = _density_curve(trunc, x_max=hi, bins=180, smooth_window=7)
    plt.plot(x_raw, y_raw, linewidth=1.6, color=raw_color, label="Raw ratio")
    plt.fill_between(x_tr, 0.0, y_tr, color="tab:orange", alpha=0.30, label="Truncated at 1")
    plt.plot(x_tr, y_tr, linewidth=1.2, color="tab:orange")
    plt.axvline(1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.9, label="R=1")
    plt.xlim(0.0, hi)
    plt.xlabel("Ratio R = pi(u|s) / mu(u|s)")
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_ratio_distributions_by_advantage(
    ratios: np.ndarray,
    advantages: np.ndarray,
    out_dir: Path,
) -> dict[str, int]:
    ratios = np.asarray(ratios)
    advantages = np.asarray(advantages)
    finite_nonneg = ratios[np.isfinite(ratios)]
    finite_nonneg = finite_nonneg[finite_nonneg >= 0.0]
    x_max = float(np.quantile(finite_nonneg, 0.995)) if finite_nonneg.size > 0 else 1.0
    x_max = max(x_max, 1.05)

    pos_mask = advantages > 0.0
    neg_mask = advantages < 0.0
    pos_raw = ratios[pos_mask]
    neg_raw = ratios[neg_mask]

    _plot_ratio_hist(
        ratios=ratios,
        out_path=out_dir / "ratio_distribution_total.png",
        title="Total Ratio Distribution",
        histtype="step",
        alpha=1.0,
        color="tab:blue",
        x_max=x_max,
    )
    _plot_ratio_hist(
        ratios=ratios,
        out_path=out_dir / "ratio_distribution_filled.png",
        title="Total Ratio Distribution (Filled)",
        histtype="stepfilled",
        alpha=0.55,
        color="tab:blue",
        x_max=x_max,
    )
    _plot_ratio_hist(
        ratios=pos_raw,
        out_path=out_dir / "ratio_distribution_adv_positive.png",
        title="Ratio Distribution (Advantage > 0)",
        histtype="stepfilled",
        alpha=0.55,
        color="tab:green",
        x_max=x_max,
    )
    _plot_ratio_hist(
        ratios=neg_raw,
        out_path=out_dir / "ratio_distribution_adv_negative.png",
        title="Ratio Distribution (Advantage < 0)",
        histtype="stepfilled",
        alpha=0.55,
        color="tab:red",
        x_max=x_max,
    )

    plot_truncation_overlay_at_one(
        raw_ratios=pos_raw,
        trunc_ratios=np.minimum(pos_raw, 1.0),
        out_path=out_dir / "ratio_distribution_adv_positive_trunc1_overlay.png",
        title="Advantage > 0: Raw vs Truncated-at-1 Ratio",
        raw_color="tab:green",
    )
    plot_truncation_overlay_at_one(
        raw_ratios=neg_raw,
        trunc_ratios=np.maximum(neg_raw, 1.0),
        out_path=out_dir / "ratio_distribution_adv_negative_trunc1_overlay.png",
        title="Advantage < 0: Raw vs Truncated-at-1 Ratio",
        raw_color="tab:red",
    )

    return {
        "adv_positive_count": int(np.sum(pos_mask)),
        "adv_negative_count": int(np.sum(neg_mask)),
        "adv_zero_count": int(np.sum(~(pos_mask | neg_mask))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1D LQG toy RL: behavior sweep + TV-constraint sweep")

    parser.add_argument("--output-dir", type=Path, default=Path("toy_lqg_ratio_tv/outputs"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--T", type=int, default=50_000)
    parser.add_argument("--burnin", type=int, default=2_000)

    parser.add_argument("--k-points", type=int, default=11, help="Number of sparse sweep points for policy gain k.")
    parser.add_argument("--k-min", type=float, default=None, help="Optional lower bound for k sweep.")
    parser.add_argument("--k-max", type=float, default=None, help="Optional upper bound for k sweep.")

    parser.add_argument("--mu-k", type=float, default=0.0)
    parser.add_argument("--mu-k-list", type=str, default=None, help="Comma-separated behavior gains, e.g. '0.0,0.4'.")
    parser.add_argument("--mu-sigma", type=float, default=1.0)
    parser.add_argument("--pi-sigma", type=float, default=None, help="If omitted, pi sigma is set equal to mu sigma.")

    parser.add_argument("--tv-constraint", type=float, default=None, help="Optional single TV constraint to include.")
    parser.add_argument(
        "--tv-constraints",
        type=str,
        default=None,
        help="Comma-separated TV constraints to sweep, e.g. '0.0,0.05,0.1'.",
    )
    parser.add_argument(
        "--tv-constraint-points",
        type=int,
        default=6,
        help="If tv-constraints is not set, sweep this many points from 0 to max TV in k sweep.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Deprecated alias: if set, includes TV constraint epsilon/2.",
    )

    parser.add_argument("--s0", type=float, default=0.0)
    parser.add_argument("--adv-state", type=float, default=1.0)
    parser.add_argument("--make-plots", action="store_true")

    parser.add_argument("--a", type=float, default=0.95)
    parser.add_argument("--b", type=float, default=0.5)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--r", type=float, default=0.1)
    parser.add_argument("--sigma-xi", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    env = LQG1D(
        a=args.a,
        b=args.b,
        q=args.q,
        r=args.r,
        sigma_xi=args.sigma_xi,
        gamma=args.gamma,
    )
    k_star, P_star = lqr_gain_1d(env.a, env.b, env.q, env.r, env.gamma)

    mu_k_values = parse_float_list(args.mu_k_list) if args.mu_k_list is not None else [float(args.mu_k)]
    if not mu_k_values:
        mu_k_values = [float(args.mu_k)]

    top_summary: dict[str, object] = {
        "lqr": {"k_star": k_star, "P_star": P_star},
        "behaviors": [],
    }

    for bidx, mu_k in enumerate(mu_k_values):
        mu = LinearGaussianPolicy(k=float(mu_k), sigma=args.mu_sigma)
        pi_sigma = mu.sigma if args.pi_sigma is None else float(args.pi_sigma)
        pi_star = LinearGaussianPolicy(k=k_star, sigma=pi_sigma)

        bdir = out / f"behavior_{bidx:02d}_mu_k_{mu.k:+.3f}"
        bdir.mkdir(parents=True, exist_ok=True)

        states, actions = collect_behavior_samples(
            env=env,
            mu=mu,
            T=args.T,
            burnin=args.burnin,
            s0=args.s0,
            seed=args.seed + bidx,
        )

        ratios_star = ratios_from_samples(pi_star, mu, states, actions)
        eabs_star = estimate_E_abs_R_minus_1(ratios_star)
        tv_star = 0.5 * eabs_star

        k_grid = build_sparse_k_grid(mu.k, k_star, args.k_points, args.k_min, args.k_max)
        sweep_rows: list[dict[str, float | bool | None]] = []
        all_ratios: list[np.ndarray] = []
        stable_rows_for_plot: list[tuple[float, float, float, float, np.ndarray]] = []

        for k in k_grid:
            pi_k = LinearGaussianPolicy(k=float(k), sigma=pi_sigma)
            ratios_k = ratios_from_samples(pi_k, mu, states, actions)
            eabs_k = estimate_E_abs_R_minus_1(ratios_k)
            tv_k = 0.5 * eabs_k

            stable = True
            try:
                P_k, C_k = value_coefficients_linear_policy(env, pi_k)
                value_s0 = -P_k * (args.s0**2) + C_k
            except ValueError:
                stable = False
                P_k, C_k, value_s0 = None, None, None

            row: dict[str, float | bool | None] = {
                "k": float(k),
                "stable": bool(stable),
                "P": float(P_k) if P_k is not None else None,
                "C": float(C_k) if C_k is not None else None,
                "value_s0": float(value_s0) if value_s0 is not None else None,
                "E_abs_R_minus_1": float(eabs_k),
                "TV": float(tv_k),
            }
            sweep_rows.append(row)
            all_ratios.append(ratios_k)
            if stable:
                stable_rows_for_plot.append((float(tv_k), float(eabs_k), float(value_s0), float(k), ratios_k))

        stable_indices = [i for i, row in enumerate(sweep_rows) if bool(row["stable"])]
        if not stable_indices:
            raise RuntimeError(f"No stable policy found for behavior mu_k={mu.k:.4f}.")

        best_unconstrained_idx = max(stable_indices, key=lambda i: float(sweep_rows[i]["value_s0"]))
        min_tv_idx = min(stable_indices, key=lambda i: float(sweep_rows[i]["TV"]))
        max_tv_stable = max(float(sweep_rows[i]["TV"]) for i in stable_indices)

        tv_constraints = build_tv_constraint_values(
            max_tv=max_tv_stable,
            tv_constraints_raw=args.tv_constraints,
            tv_constraint_single=args.tv_constraint,
            epsilon=args.epsilon,
            tv_constraint_points=args.tv_constraint_points,
        )

        selected_ratio_arrays: list[np.ndarray] = []
        selected_tvs: list[float] = []
        selected_eabs: list[float] = []
        selected_ks: list[float] = []
        selected_values: list[float] = []
        constraint_rows: list[dict[str, float | int | bool]] = []

        for cidx, tv_c in enumerate(tv_constraints):
            feasible = [i for i in stable_indices if float(sweep_rows[i]["TV"]) <= float(tv_c) + 1e-12]
            fallback_used = len(feasible) == 0
            chosen_idx = max(feasible, key=lambda i: float(sweep_rows[i]["value_s0"])) if feasible else min_tv_idx
            chosen_row = sweep_rows[chosen_idx]

            chosen_k = float(chosen_row["k"])
            chosen_tv = float(chosen_row["TV"])
            chosen_val = float(chosen_row["value_s0"])
            chosen_ratios = all_ratios[chosen_idx]
            chosen_eabs = estimate_E_abs_R_minus_1(chosen_ratios)

            chosen_pi = LinearGaussianPolicy(k=chosen_k, sigma=pi_sigma)
            P_sel, C_sel = value_coefficients_linear_policy(env, chosen_pi)
            chosen_adv = advantage_under_linear_policy(env, P_sel, C_sel, states, actions)
            adv_counts = {
                "adv_positive_count": int(np.sum(chosen_adv > 0.0)),
                "adv_negative_count": int(np.sum(chosen_adv < 0.0)),
                "adv_zero_count": int(np.sum(chosen_adv == 0.0)),
            }

            selected_ratio_arrays.append(chosen_ratios)
            selected_tvs.append(chosen_tv)
            selected_eabs.append(chosen_eabs)
            selected_ks.append(chosen_k)
            selected_values.append(chosen_val)

            constraint_rows.append(
                {
                    "tv_constraint": float(tv_c),
                    "selected_k": chosen_k,
                    "selected_tv": chosen_tv,
                    "selected_value_s0": chosen_val,
                    "feasible_count": int(len(feasible)),
                    "fallback_used": bool(fallback_used),
                    **adv_counts,
                }
            )

            if args.make_plots:
                cdir = bdir / f"constraint_{cidx:02d}_tv_{float(tv_c):.4f}"
                cdir.mkdir(parents=True, exist_ok=True)
                plot_ratio_distributions_by_advantage(chosen_ratios, chosen_adv, cdir)

        stable_rows_for_plot = sorted(stable_rows_for_plot, key=lambda t: t[0])
        stable_tvs = np.asarray([x[0] for x in stable_rows_for_plot], dtype=np.float64)
        stable_eabs = np.asarray([x[1] for x in stable_rows_for_plot], dtype=np.float64)
        stable_values = np.asarray([x[2] for x in stable_rows_for_plot], dtype=np.float64)
        stable_ks = np.asarray([x[3] for x in stable_rows_for_plot], dtype=np.float64)
        stable_ratio_arrays = [x[4] for x in stable_rows_for_plot]

        selected_tvs_arr = np.asarray(selected_tvs, dtype=np.float64)
        selected_eabs_arr = np.asarray(selected_eabs, dtype=np.float64)
        selected_ks_arr = np.asarray(selected_ks, dtype=np.float64)
        selected_vals_arr = np.asarray(selected_values, dtype=np.float64)

        np.savez_compressed(
            bdir / "k_sweep.npz",
            k_grid=np.asarray([float(row["k"]) for row in sweep_rows], dtype=np.float64),
            stable_mask=np.asarray([bool(row["stable"]) for row in sweep_rows], dtype=bool),
            tv_by_k=np.asarray([float(row["TV"]) for row in sweep_rows], dtype=np.float64),
            eabs_by_k=np.asarray([float(row["E_abs_R_minus_1"]) for row in sweep_rows], dtype=np.float64),
            value_s0_by_k=np.asarray(
                [np.nan if row["value_s0"] is None else float(row["value_s0"]) for row in sweep_rows],
                dtype=np.float64,
            ),
            tv_constraints=tv_constraints,
            selected_k_by_constraint=selected_ks_arr,
            selected_tv_by_constraint=selected_tvs_arr,
            selected_value_by_constraint=selected_vals_arr,
            ratios_star=ratios_star,
        )

        metrics: dict[str, object] = {
            "config": {
                "seed": int(args.seed + bidx),
                "T": args.T,
                "burnin": args.burnin,
                "epsilon": args.epsilon,
                "env": {
                    "a": env.a,
                    "b": env.b,
                    "q": env.q,
                    "r": env.r,
                    "sigma_xi": env.sigma_xi,
                    "gamma": env.gamma,
                },
                "mu": {"k": mu.k, "sigma": mu.sigma},
                "pi_star": {"k": pi_star.k, "sigma": pi_star.sigma},
            },
            "lqr": {"k_star": k_star, "P_star": P_star},
            "star": {"E_abs_R_minus_1": eabs_star, "TV": tv_star},
            "sweep_setup": {
                "k_points": int(args.k_points),
                "k_min": float(k_grid[0]),
                "k_max": float(k_grid[-1]),
                "tv_constraints": [float(x) for x in tv_constraints],
            },
            "k_sweep": sweep_rows,
            "best_unconstrained": {
                "k": float(sweep_rows[best_unconstrained_idx]["k"]),
                "value_s0": float(sweep_rows[best_unconstrained_idx]["value_s0"]),
                "TV": float(sweep_rows[best_unconstrained_idx]["TV"]),
            },
            "best_by_tv_constraint": constraint_rows,
        }

        with (bdir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        if args.make_plots:
            try:
                plot_ratio_hist_overlay(selected_tvs_arr, selected_ratio_arrays, bdir / "ratio_hist_selected_by_tv_constraint.png")
                plot_ratio_quantiles_vs_tv(
                    selected_tvs_arr,
                    selected_ratio_arrays,
                    bdir / "ratio_quantiles_selected_by_tv_constraint.png",
                )
                plot_tv_identity(
                    selected_tvs_arr,
                    selected_eabs_arr,
                    bdir / "tv_identity_selected_by_tv_constraint.png",
                )
                plot_selected_k_vs_tv_constraint(
                    tv_constraints,
                    selected_ks_arr,
                    selected_tvs_arr,
                    bdir / "selected_k_vs_tv_constraint.png",
                )
                plot_value_vs_tv(
                    stable_tvs,
                    stable_values,
                    stable_ks,
                    float(tv_constraints[len(tv_constraints) // 2]),
                    bdir / "value_vs_tv_sparse_k_sweep.png",
                )
                plot_ratio_hist_overlay(stable_tvs, stable_ratio_arrays, bdir / "ratio_hist_all_k_sweep.png")
                plot_ratio_quantiles_vs_tv(stable_tvs, stable_ratio_arrays, bdir / "ratio_quantiles_all_k_sweep.png")
                plot_tv_identity(stable_tvs, stable_eabs, bdir / "tv_identity_all_k_sweep.png")
                plot_advantage_curve(env, P_star, args.adv_state, bdir / "advantage_curve.png")
            except Exception as exc:
                print(f"Plotting skipped for mu_k={mu.k:.4f} due to error: {exc}")

        top_summary["behaviors"].append(
            {
                "behavior_dir": str(bdir),
                "mu_k": float(mu.k),
                "best_unconstrained_k": float(sweep_rows[best_unconstrained_idx]["k"]),
                "best_unconstrained_tv": float(sweep_rows[best_unconstrained_idx]["TV"]),
                "num_tv_constraints": int(len(tv_constraints)),
            }
        )

        print(f"Behavior mu_k={mu.k:.3f} -> {bdir}")
        print(
            f"  Best unconstrained: k={float(sweep_rows[best_unconstrained_idx]['k']):.4f}, "
            f"TV={float(sweep_rows[best_unconstrained_idx]['TV']):.4f}, "
            f"value={float(sweep_rows[best_unconstrained_idx]['value_s0']):.4f}"
        )
        if constraint_rows:
            c0 = constraint_rows[0]
            c1 = constraint_rows[-1]
            print(
                f"  TV constraints: [{float(c0['tv_constraint']):.4f}, {float(c1['tv_constraint']):.4f}] "
                f"({len(constraint_rows)} points)"
            )

    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(top_summary, f, indent=2)

    print(f"Top-level summary: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
