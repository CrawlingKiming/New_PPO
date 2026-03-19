#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TAG="$(date +"%Y%m%d_%H%M%S")"
RESULTS_DIR="${ROOT_DIR}/results/mujoco_compare_${RUN_TAG}"

ALGO_A="opspo"
ALGO_B="spo"

MUJOCO_ENVS="HalfCheetah-v4" #"Humanoid-v4"
MUJOCO_SEEDS="1,2,3"
EPSILON=""
GPD_SHAPE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --algo-a)
      ALGO_A="$2"
      shift 2
      ;;
    --algo-b)
      ALGO_B="$2"
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
    --epsilon)
      EPSILON="$2"
      shift 2
      ;;
    --gpd-shape|--gpd_shape)
      GPD_SHAPE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${RESULTS_DIR}"

echo "Results dir: ${RESULTS_DIR}"
echo "Algorithms: ${ALGO_A} vs ${ALGO_B}"
[[ -n "${EPSILON}" ]] && echo "Epsilon override: ${EPSILON}"
[[ -n "${GPD_SHAPE}" ]] && echo "GPD shape override: ${GPD_SHAPE}"

run_mujoco() {
  local algo="$1"
  echo "Running MuJoCo ${algo}..."
  cmd=(
    conda run -n opo-mujoco python "${ROOT_DIR}/mujoco/main.py"
    --algo "${algo}" --envs "${MUJOCO_ENVS}" --seeds "${MUJOCO_SEEDS}"
  )
  [[ -n "${EPSILON}" ]] && cmd+=(--epsilon "${EPSILON}")
  [[ -n "${GPD_SHAPE}" ]] && cmd+=(--gpd_shape "${GPD_SHAPE}")
  (cd "${RESULTS_DIR}" && "${cmd[@]}")
}

for algo in "${ALGO_A}" "${ALGO_B}"; do
  run_mujoco "${algo}"
done

conda run -n opo-mujoco python "${ROOT_DIR}/scripts/collect_results.py" \
  --root "${RESULTS_DIR}" --out "${RESULTS_DIR}/mujoco_results.json"

python "${ROOT_DIR}/scripts/aggregate_results.py" \
  --input "${RESULTS_DIR}/mujoco_results.json" --algo-a "${ALGO_A}" --algo-b "${ALGO_B}" \
  --out "${RESULTS_DIR}/mujoco_compare.csv"

echo "Done."
