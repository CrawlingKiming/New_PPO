# Toy 1D LQG Ratio/TV Example

This folder is a self-contained toy RL simulation for:
- 1D LQG dynamics and quadratic reward.
- LQR-based `k*` as an unconstrained reference policy.
- Importance ratio `R = pi(u|s) / mu(u|s)` under behavior samples.
- Sparse sweep over policy gains `k`.
- TV-constrained selection by `k`-sweep (no mixture-policy shortcut).
- Multiple behavior policies (`mu`) in one run.
- Ratio distributions split by advantage sign, including explicit truncation-at-1 overlays.

## Run

```bash
python toy_lqg_ratio_tv/lqg_ratio_tv.py \
  --mu-k-list '0.0,0.4' \
  --k-points 9 \
  --tv-constraints '0.0,0.08,0.16,0.24' \
  --make-plots \
  --output-dir toy_lqg_ratio_tv/outputs
```

Notes:
- If `--mu-k-list` is omitted, it uses `--mu-k`.
- If `--tv-constraints` is omitted, constraints are auto-swept from `0` to max TV seen in the `k` sweep.
- `--epsilon` is a backward-compatible alias that adds a constraint at `epsilon/2`.
- By default, `sigma_pi = sigma_mu`; override with `--pi-sigma`.

## Output Layout

For each behavior policy, files are written to:
- `outputs/behavior_XX_mu_k_.../`

Key files per behavior:
- `metrics.json`: full sweep results and best policy per TV constraint.
- `k_sweep.npz`: arrays from the sparse `k` sweep and selected policies.
- `ratio_hist_selected_by_tv_constraint.png`: ratio overlays across selected constrained policies.
- `selected_k_vs_tv_constraint.png`: selected `k` and realized TV vs constraint.
- `value_vs_tv_sparse_k_sweep.png`: value vs TV across all swept `k`.

For each TV constraint (subfolder `constraint_XX_tv_.../`):
- `ratio_distribution_total.png`
- `ratio_distribution_filled.png`
- `ratio_distribution_adv_positive.png`
- `ratio_distribution_adv_negative.png`
- `ratio_distribution_adv_positive_trunc1_overlay.png`
- `ratio_distribution_adv_negative_trunc1_overlay.png`

The two `*_trunc1_overlay.png` plots explicitly show truncation at `R=1`:
- Advantage > 0: compare raw `R` vs `min(R, 1)`.
- Advantage < 0: compare raw `R` vs `max(R, 1)`.
