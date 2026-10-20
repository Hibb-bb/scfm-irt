#!/usr/bin/env bash
# Submit human thymus evaluation + 4-dataset rollout.
#
#   bash scripts/submit_thymus_rollout.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

echo "=== human_thymus download (Census) ==="
THYMUS_DL="$(sbatch --parsable slurm/download_human_thymus.slurm)"
echo "thymus-dl=$THYMUS_DL"

echo "=== human_thymus v1 extractions ==="
THYMUS_EXTRACT="$(sbatch --parsable --dependency="afterok:$THYMUS_DL" slurm/human_thymus_v1_extract.slurm)"
echo "thymus-v1-extract=$THYMUS_EXTRACT"

echo "=== State on human_thymus ==="
STATE_THYMUS="$(sbatch --parsable \
  --dependency="afterok:$THYMUS_EXTRACT" \
  --export="ALL,DATASET=human_thymus,SPLIT=cross_donor" \
  --job-name=state-thymus \
  slurm/state_redo.slurm)"
echo "state-thymus=$STATE_THYMUS"

echo "=== human_thymus v1 ranking ==="
THYMUS_RANK="$(sbatch --parsable \
  --dependency="afterok:$THYMUS_EXTRACT,$STATE_THYMUS" \
  slurm/human_thymus_v1.slurm)"
echo "thymus-v1-rank=$THYMUS_RANK"

echo "=== 4-dataset rollout report ==="
ROLLOUT="$(sbatch --parsable \
  --dependency="afterok:$THYMUS_RANK" \
  slurm/rollout_report.slurm)"
echo "rollout-report=$ROLLOUT"

echo
echo "Submitted thymus pipeline."
echo "  thymus-dl:         $THYMUS_DL"
echo "  thymus-v1-extract: $THYMUS_EXTRACT"
echo "  state-thymus:      $STATE_THYMUS"
echo "  thymus-v1-rank:    $THYMUS_RANK"
echo "  rollout-report:    $ROLLOUT"
