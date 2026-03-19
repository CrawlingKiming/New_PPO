#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TAG="$(date +"%Y%m%d_%H%M%S")"
ALGO="opspo_fixed"
RESULTS_ROOT="${ROOT_DIR}/results/mujoco_${ALGO}_${RUN_TAG}"
RESULTS_ROOT_EXPLICIT=0

MUJOCO_ENVS="Ant-v4,HalfCheetah-v4,Hopper-v4,Humanoid-v4,HumanoidStandup-v4,Walker2d-v4"
MUJOCO_SEEDS="1,2,3"
EPSILON=""
GPD_SHAPE=""
MINI_BATCHES=""

CONDA_ENV="opo-mujoco"
TIME_LIMIT="14:00:00"
CPUS_PER_TASK="8"
MEM_GB="16G"
ACCOUNT="laberlabs"
PARTITION="gpu-common"
GPU_GRES="gpu:2080:1"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_opspo_fixed_mujoco_slurm.sh [options]

Options:
  --algo <name>              Training algorithm (default: opspo_fixed).
  --results-root <path>      Root output dir for this batch of jobs.
  --mujoco-envs <csv>        Comma-separated env list.
  --mujoco-seeds <csv>       Comma-separated seed list.
  --epsilon <float>          PPO clip epsilon override.
  --gpd-shape <float>        GPD shape override for OPSPO variants.
  --mini-batches <int>       Override number of minibatches.
  --conda-env <name>         Conda env for training (default: opo-mujoco).
  --time <HH:MM:SS>          Slurm time limit (default: 24:00:00).
  --cpus <int>               CPUs per task (default: 8).
  --mem <size>               Memory per task (default: 32G).
  --partition <name>         Slurm partition (optional).
  --account <name>           Slurm account (optional).
  --gpu-gres <gres>          Slurm GRES string (default: gpu:2080:1).
  -h, --help                 Show this help text.

Example:
  bash scripts/submit_opspo_fixed_mujoco_slurm.sh \
    --mujoco-envs "Ant-v4,HalfCheetah-v4,Hopper-v4,Humanoid-v4,HumanoidStandup-v4,Walker2d-v4" \
    --mujoco-seeds "1,2,3" \
    --partition gpu-common \
    --account laberlabs \
    --gpu-gres gpu:2080:1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --algo)
      ALGO="$2"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="$2"
      RESULTS_ROOT_EXPLICIT=1
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
    --mini-batches|--mini_batches)
      MINI_BATCHES="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --time)
      TIME_LIMIT="$2"
      shift 2
      ;;
    --cpus)
      CPUS_PER_TASK="$2"
      shift 2
      ;;
    --mem)
      MEM_GB="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --account)
      ACCOUNT="$2"
      shift 2
      ;;
    --gpu-gres)
      GPU_GRES="$2"
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

if [[ "${RESULTS_ROOT_EXPLICIT}" -eq 0 ]]; then
  RESULTS_ROOT="${ROOT_DIR}/results/mujoco_${ALGO}_${RUN_TAG}"
fi

mkdir -p "${RESULTS_ROOT}"

IFS=',' read -r -a ENV_ARRAY <<< "${MUJOCO_ENVS}"
if [[ "${#ENV_ARRAY[@]}" -eq 0 ]]; then
  echo "No environments provided via --mujoco-envs" >&2
  exit 1
fi

echo "Submitting MuJoCo ${ALGO} jobs"
echo "Results root: ${RESULTS_ROOT}"
echo "Seeds: ${MUJOCO_SEEDS}"
echo "GPU request: ${GPU_GRES}"
[[ -n "${EPSILON}" ]] && echo "Epsilon override: ${EPSILON}"
[[ -n "${GPD_SHAPE}" ]] && echo "GPD shape override: ${GPD_SHAPE}"
[[ -n "${MINI_BATCHES}" ]] && echo "Mini-batches override: ${MINI_BATCHES}"

for raw_env in "${ENV_ARRAY[@]}"; do
  env_name="$(echo "${raw_env}" | xargs)"
  [[ -z "${env_name}" ]] && continue

  env_results_dir="${RESULTS_ROOT}/${env_name}"
  mkdir -p "${env_results_dir}"
  log_dir="${env_results_dir}/logs"
  mkdir -p "${log_dir}"

  job_name="${ALGO//_/-}-${env_name}"

  sbatch_args=(
    --job-name="${job_name}"
    --output="${log_dir}/%x_%j.out"
    --error="${log_dir}/%x_%j.err"
    --time="${TIME_LIMIT}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEM_GB}"
    --gres="${GPU_GRES}"
  )

  if [[ -n "${PARTITION}" ]]; then
    sbatch_args+=(--partition="${PARTITION}")
  fi
  if [[ -n "${ACCOUNT}" ]]; then
    sbatch_args+=(--account="${ACCOUNT}")
  fi

  extra_args=""
  if [[ -n "${EPSILON}" ]]; then
    extra_args+=" --epsilon \"${EPSILON}\""
  fi
  if [[ -n "${GPD_SHAPE}" ]]; then
    extra_args+=" --gpd-shape \"${GPD_SHAPE}\""
  fi
  if [[ -n "${MINI_BATCHES}" ]]; then
    extra_args+=" --mini-batches \"${MINI_BATCHES}\""
  fi

  sbatch "${sbatch_args[@]}" --wrap \
    "set -euo pipefail; \
     cd \"${ROOT_DIR}\"; \
     bash \"${ROOT_DIR}/scripts/run_opspo_fixed_mujoco.sh\" \
       --algo \"${ALGO}\" \
       --results-dir \"${env_results_dir}\" \
       --mujoco-envs \"${env_name}\" \
       --mujoco-seeds \"${MUJOCO_SEEDS}\" \
       --conda-env \"${CONDA_ENV}\"${extra_args}"
done

echo "All jobs submitted."
