#!/usr/bin/env bash
# Submit the full rollout expansion pipeline (no scFoundation).
#
#   bash scripts/submit_rollout_expansion.sh
#
# Chain:
#   1. State re-extract (pbmc3k, lung_atlas, immune_human) — normalize+log1p fix
#   2. immune_human v5 extractions (missing embedders)
#   3. pbmc3k v3 ranking (after state-pbmc3k)
#   4. lung v5 re-rank (after state-lung)
#   5. immune_human v5 ranking (after extractions + state-immune)
#   6. rollout simulation report (after all three ranking runs)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

job_id() {
  squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk -v n="$1" '$2 ~ n { print $1; exit }'
}

submit_state() {
  local dataset=$1 split=$2 pattern=$3
  local existing
  existing="$(job_id "$pattern")"
  if [[ -n "$existing" ]]; then
    echo "[state] reusing queued job $pattern=$existing" >&2
    echo "$existing"
    return
  fi
  sbatch --parsable \
    --export="ALL,DATASET=$dataset,SPLIT=$split" \
    --job-name="$pattern" \
    slurm/state_redo.slurm
}

echo "=== State re-extract (normalize+log1p) ==="
STATE_PBMC="$(submit_state pbmc3k stratified state-pb)"
STATE_LUNG="$(submit_state lung_atlas cross_donor state-lung)"

echo "=== immune_human v5 extractions ==="
IMMUNE_EXTRACT="$(sbatch --parsable slurm/immune_human_v5_extract.slurm)"
echo "immune-v5-extract=$IMMUNE_EXTRACT"

echo "=== State on immune_human (after other immune extractions) ==="
STATE_IMMUNE="$(sbatch --parsable \
  --dependency="afterok:$IMMUNE_EXTRACT" \
  --export="ALL,DATASET=immune_human,SPLIT=cross_donor" \
  --job-name=state-im \
  slurm/state_redo.slurm)"
echo "state-immune=$STATE_IMMUNE"

echo "=== Ranking runs ==="
PBMC_V3="$(sbatch --parsable --dependency="afterok:$STATE_PBMC" slurm/pbmc3k_v3.slurm)"
echo "pbmc3k-v3=$PBMC_V3"

LUNG_V5="$(sbatch --parsable --dependency="afterok:$STATE_LUNG" slurm/lung_atlas_v5.slurm)"
echo "lung-v5-rerun=$LUNG_V5"

IMMUNE_V5="$(sbatch --parsable \
  --dependency="afterok:$IMMUNE_EXTRACT,$STATE_IMMUNE" \
  slurm/immune_human_v5.slurm)"
echo "immune-v5-rank=$IMMUNE_V5"

echo "=== Rollout simulation (3 datasets) ==="
ROLLOUT="$(sbatch --parsable \
  --dependency="afterok:$PBMC_V3,$LUNG_V5,$IMMUNE_V5" \
  slurm/rollout_report.slurm)"
echo "rollout-report=$ROLLOUT"

echo
echo "Submitted pipeline (scFoundation excluded)."
echo "  state-pbmc3k:      $STATE_PBMC"
echo "  state-lung:        $STATE_LUNG"
echo "  immune-v5-extract: $IMMUNE_EXTRACT"
echo "  state-immune:      $STATE_IMMUNE"
echo "  pbmc3k-v3:         $PBMC_V3"
echo "  lung-v5-rerun:     $LUNG_V5"
echo "  immune-v5-rank:    $IMMUNE_V5"
echo "  rollout-report:    $ROLLOUT"
