#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TAG="$(date +"%Y%m%d_%H%M%S")"
RESULTS_DIR="${ROOT_DIR}/results/atari_compare_${RUN_TAG}"

ALGO_A="opo"
ALGO_B="spo"

ATARI_ENVS="Assault,Asterix,BeamRider,SpaceInvaders"
ATARI_SEEDS="1,2,3"

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
    --atari-envs)
      ATARI_ENVS="$2"
      shift 2
      ;;
    --atari-seeds)
      ATARI_SEEDS="$2"
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

run_atari() {
  local algo="$1"
  echo "Running Atari ${algo}..."
  (cd "${RESULTS_DIR}" && \
    conda run -n opo-atari python "${ROOT_DIR}/atari/main.py" \
      --algo "${algo}" --envs "${ATARI_ENVS}" --seeds "${ATARI_SEEDS}")
}

for algo in "${ALGO_A}" "${ALGO_B}"; do
  run_atari "${algo}"
done

conda run -n opo-atari python "${ROOT_DIR}/scripts/collect_results.py" \
  --root "${RESULTS_DIR}" --out "${RESULTS_DIR}/atari_results.json"

python "${ROOT_DIR}/scripts/aggregate_results.py" \
  --input "${RESULTS_DIR}/atari_results.json" --algo-a "${ALGO_A}" --algo-b "${ALGO_B}" \
  --out "${RESULTS_DIR}/atari_compare.csv"

echo "Done."
