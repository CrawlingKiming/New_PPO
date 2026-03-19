# OPO

This repository contains policy optimization experiments built around the MuJoCo training stack in `mujoco/`. The main code supports PPO, SPO, OPO, OPSPO, and several fixed-threshold and penalty-based variants used for offline comparison runs and Slurm sweeps.

The current workflow is:

- Run training from `mujoco/main.py`.
- Use the wrappers in `scripts/` to launch local runs or submit MuJoCo jobs to Slurm.
- Collect TensorBoard results with `scripts/collect_results.py` and related plotting or aggregation helpers.

Common entry points:

```bash
python mujoco/main.py --algo spo
```

```bash
bash scripts/run_opspo_fixed_mujoco.sh --algo opspo_fixed --epsilon 0.1 --gpd-shape 0.49
```

```bash
bash scripts/submit_opspo_fixed_mujoco_slurm.sh --algo opspo_fixed --epsilon 0.1 --gpd-shape 0.49
```

Notes:

- MuJoCo experiments write TensorBoard event files into per-run directories under the current working directory.
- Slurm wrappers default to the `opo-mujoco` Conda environment and submit one job per environment.
- The Atari code is present in the repo, but this README is intentionally focused on the MuJoCo path.
