#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TAG="$(date +"%Y%m%d_%H%M%S")"
ALGO="opspo_fixed"
RESULTS_DIR="${ROOT_DIR}/results/mujoco_${ALGO}_${RUN_TAG}"
RESULTS_DIR_EXPLICIT=0

MUJOCO_ENVS="Ant-v4,HalfCheetah-v4,Hopper-v4,Humanoid-v4,HumanoidStandup-v4,Walker2d-v4"
MUJOCO_SEEDS="1,2,3"
CONDA_ENV="opo-mujoco"
EPSILON=""
GPD_SHAPE=""
MINI_BATCHES=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_opspo_fixed_mujoco.sh [options]

Options:
  --algo <name>             Training algorithm (default: opspo_fixed).
  --results-dir <path>      Output directory for TensorBoard logs/results.
  --mujoco-envs <csv>       Comma-separated env list (default: Ant-v4,HalfCheetah-v4,Hopper-v4,Humanoid-v4,HumanoidStandup-v4,Walker2d-v4).
  --mujoco-seeds <csv>      Comma-separated seed list (default: 1,2,3).
  --conda-env <name>        Conda environment name (default: opo-mujoco).
  --epsilon <float>         PPO clip epsilon override (optional; default from mujoco/main.py).
  --gpd-shape <float>       GPD shape override for OPSPO variants (optional; default from mujoco/main.py).
  --mini-batches <int>      Override number of minibatches (optional; default from mujoco/main.py).
  -h, --help                Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --algo)
      ALGO="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      RESULTS_DIR_EXPLICIT=1
      shift 2
      ;;
    --mujoco-envs)
      MUJOCO_ENVS="$2"
      shift 2
      ;;
    --mujoco-seeds)
      MUJOCO_SEEDS="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --epsilon)
      EPSILON="$2"
      shift 2
      ;;
    --gpd-shape|--gpd_shape)
      GPD_SHAPE="$2"
      shift 2
      ;;
    --mini-batches|--mini_batches)
      MINI_BATCHES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${RESULTS_DIR_EXPLICIT}" -eq 0 ]]; then
  RESULTS_DIR="${ROOT_DIR}/results/mujoco_${ALGO}_${RUN_TAG}"
fi

mkdir -p "${RESULTS_DIR}"

echo "Results dir: ${RESULTS_DIR}"
echo "Algorithm: ${ALGO}"
echo "MuJoCo envs: ${MUJOCO_ENVS}"
echo "MuJoCo seeds: ${MUJOCO_SEEDS}"
[[ -n "${EPSILON}" ]] && echo "Epsilon override: ${EPSILON}"
[[ -n "${GPD_SHAPE}" ]] && echo "GPD shape override: ${GPD_SHAPE}"
[[ -n "${MINI_BATCHES}" ]] && echo "Mini-batches override: ${MINI_BATCHES}"

cmd=(
  conda run -n "${CONDA_ENV}" python "${ROOT_DIR}/mujoco/main.py"
  --algo "${ALGO}"
  --envs "${MUJOCO_ENVS}"
  --seeds "${MUJOCO_SEEDS}"
)
[[ -n "${EPSILON}" ]] && cmd+=(--epsilon "${EPSILON}")
[[ -n "${GPD_SHAPE}" ]] && cmd+=(--gpd_shape "${GPD_SHAPE}")
[[ -n "${MINI_BATCHES}" ]] && cmd+=(--mini_batches "${MINI_BATCHES}")

(
  cd "${RESULTS_DIR}"
  "${cmd[@]}"
)

conda run -n "${CONDA_ENV}" python "${ROOT_DIR}/scripts/collect_results.py" \
  --root "${RESULTS_DIR}" --out "${RESULTS_DIR}/mujoco_results.json"

echo "Done."
