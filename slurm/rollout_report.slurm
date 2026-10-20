#!/bin/bash
#SBATCH --job-name=rollout-report
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:0
#SBATCH --time=12:00:00
#SBATCH --output=logs/rollout_report_%j.out
#SBATCH --error=logs/rollout_report_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs sim/results/rollout

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate

PYTHONPATH=. python sim/run_rollout_report.py
echo "[rollout] done -> sim/results/rollout/"
