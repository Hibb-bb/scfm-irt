#!/bin/bash
#SBATCH --job-name=scfm-adapt-lung
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/adaptive_calibration_lung_%j.out
#SBATCH --error=logs/adaptive_calibration_lung_%j.err

# Sample-efficient calibration sim on lung_atlas IRT run (~10k items, 5 models).
# LOO difficulty stability + new-model adaptive ability estimation.
#
#   sbatch slurm/adaptive_calibration_lung.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate

RUN="runs/ranking_lung_atlas_v1"
test -f "$RUN/ability_ranking.csv" || {
  echo "[error] missing IRT run at $RUN — run atlas_lung first"
  exit 1
}

echo "[adaptive-lung] host=$(hostname)  run=$RUN"
python scripts/adaptive_calibration_sim.py \
  --run "$RUN" \
  --replicates 40 \
  --steps 350 \
  --seed 0

echo "[adaptive-lung] done -> $RUN/adaptive_calibration_sim/report.md"
