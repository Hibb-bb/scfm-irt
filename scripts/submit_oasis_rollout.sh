#!/usr/bin/env bash
# Submit OASIS evaluation + 4-dataset rollout (after staging h5ad).
#
# Prerequisites:
#   1. NBDC/JGA approved OASIS h5ad staged at data/staging/oasis.h5ad
#      OR export OASIS_H5AD_PATH=/path/to/oasis.h5ad
#   2. Existing lung/pbmc/immune ranking runs (sim/conf/sim.yaml)
#
#   bash scripts/submit_oasis_rollout.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs data/staging

if [[ ! -f data/raw/oasis.h5ad ]] && [[ -z "${OASIS_H5AD_PATH:-}" ]]; then
  if ! compgen -G "data/staging/oasis*.h5ad" > /dev/null && \
     ! compgen -G "data/staging/OASIS*.h5ad" > /dev/null; then
    echo "[error] No OASIS h5ad staged." >&2
    echo "  OASIS requires controlled access (NBDC hum0197 / JGA JGAS000783)." >&2
    echo "  Stage processed h5ad at data/staging/oasis.h5ad or set OASIS_H5AD_PATH." >&2
    exit 1
  fi
fi

echo "=== OASIS download / normalize ==="
OASIS_DL="$(sbatch --parsable slurm/download_oasis.slurm)"
echo "oasis-dl=$OASIS_DL"

echo "=== OASIS v1 extractions ==="
OASIS_EXTRACT="$(sbatch --parsable --dependency="afterok:$OASIS_DL" slurm/oasis_v1_extract.slurm)"
echo "oasis-v1-extract=$OASIS_EXTRACT"

echo "=== State on OASIS ==="
STATE_OASIS="$(sbatch --parsable \
  --dependency="afterok:$OASIS_EXTRACT" \
  --export="ALL,DATASET=oasis,SPLIT=cross_donor" \
  --job-name=state-oasis \
  slurm/state_redo.slurm)"
echo "state-oasis=$STATE_OASIS"

echo "=== OASIS v1 ranking ==="
OASIS_RANK="$(sbatch --parsable \
  --dependency="afterok:$OASIS_EXTRACT,$STATE_OASIS" \
  slurm/oasis_v1.slurm)"
echo "oasis-v1-rank=$OASIS_RANK"

echo "=== 4-dataset rollout report ==="
ROLLOUT="$(sbatch --parsable \
  --dependency="afterok:$OASIS_RANK" \
  slurm/rollout_report.slurm)"
echo "rollout-report=$ROLLOUT"

echo
echo "Submitted OASIS pipeline."
echo "  oasis-dl:         $OASIS_DL"
echo "  oasis-v1-extract: $OASIS_EXTRACT"
echo "  state-oasis:      $STATE_OASIS"
echo "  oasis-v1-rank:    $OASIS_RANK"
echo "  rollout-report:   $ROLLOUT"
