#!/usr/bin/env bash
# Submit the extraction matrix for the newer embedders.
#
# pbmc3k (stratified) is the smoke test: small, fast, full transcriptome. Only submit the
# lung_atlas (cross_donor) jobs for models that already produced a pbmc3k cache, so a broken
# model fails on a 3k-cell dataset instead of burning a GPU day on 50k cells.
#
#   bash scripts/submit_extractions.sh pbmc3k
#   bash scripts/submit_extractions.sh lung_atlas
#   bash scripts/submit_extractions.sh immune_human
#   bash scripts/submit_extractions.sh pbmc3k --dry-run
#
# Models whose assets are missing are skipped with a warning rather than submitted to fail.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET="${1:-pbmc3k}"
DRY_RUN="${2:-}"

# Resources are sized per dataset, not left at the job script's lung_atlas-shaped defaults.
# pbmc3k is 2,638 cells and finishes in minutes; asking for 12h/128G there just makes the job
# ineligible for backfill and leaves it sitting behind the whole queue.
case "$DATASET" in
  pbmc3k)
    SPLIT="stratified"
    TIME_LIMIT="01:00:00"
    MEM="48G"
    ;;
  lung_atlas)
    SPLIT="cross_donor"
    TIME_LIMIT="12:00:00"
    MEM="128G"
    ;;
  immune_human)
    SPLIT="cross_donor"
    TIME_LIMIT="12:00:00"
    MEM="128G"
    ;;
  *)
    echo "usage: $0 pbmc3k|lung_atlas|immune_human [--dry-run]"
    exit 1
    ;;
esac

MODELS=(
  geneformer_v2 cellplm transcriptformer scprint scprint2 langcell state
  geneformer_v2_316m scprint_small_v1 scprint_medium_v1 scprint_large_v1
)

# Asset each model needs before it can run at all.
assets_ready() {
  case "$1" in
    geneformer_v2)
      [[ -d cache/geneformer_workdir/Geneformer/Geneformer-V2-104M ]] ;;
    geneformer_v2_316m)
      [[ -d cache/geneformer_workdir/Geneformer/Geneformer-V2-316M ]] ;;
    scprint_small_v1)
      [[ -f cache/scprint_workdir/small-v1.ckpt ]] ;;
    scprint_medium_v1)
      [[ -f cache/scprint_workdir/medium-v1.ckpt ]] ;;
    scprint_large_v1)
      [[ -f cache/scprint_workdir/large-v1.ckpt ]] ;;
    cellplm)
      compgen -G "cache/cellplm_workdir/ckpt/*.best.ckpt" > /dev/null ;;
    transcriptformer)
      [[ -d cache/tf_workdir/checkpoints/tf_sapiens ]] ;;
    scprint)
      [[ -f cache/scprint_workdir/medium-v1.5.ckpt ]] ;;
    scprint2)
      [[ -f cache/scprint_workdir/small-v2.ckpt ]] ;;
    scfoundation)
      [[ -f cache/scfoundation_workdir/scFoundation/model/models/models.ckpt ]] ;;
    langcell)
      [[ -f cache/langcell_workdir/langcell/cell_bert/config.json ]] ;;
    state)
      [[ -f cache/state_workdir/SE-600M/se600m_epoch16.ckpt ]] ;;
    *) return 1 ;;
  esac
}

venv_for() {
  case "$1" in
    geneformer_v2*) echo ".venv-geneformer" ;;
    scprint*) echo ".venv-scprint" ;;
    state) echo ".venv-state" ;;
    *) echo ".venv-$1" ;;
  esac
}

submitted=0
skipped=()

for model in "${MODELS[@]}"; do
  venv="$(venv_for "$model")"

  if [[ ! -x "$venv/bin/python" ]]; then
    skipped+=("$model (no $venv)")
    continue
  fi
  if ! assets_ready "$model"; then
    skipped+=("$model (assets missing)")
    continue
  fi
  # For lung_atlas / immune_human, require pbmc3k smoke test first.
  if [[ "$DATASET" == "lung_atlas" || "$DATASET" == "immune_human" ]] \
    && ! compgen -G "cache/embeddings/$model/pbmc3k/*.npz" > /dev/null; then
    skipped+=("$model (no pbmc3k cache yet)")
    continue
  fi
  if compgen -G "cache/embeddings/$model/$DATASET/*.npz" > /dev/null; then
    skipped+=("$model (already cached for $DATASET)")
    continue
  fi
  # Re-running this script while jobs are still queued is normal (assets land at different
  # times); without this check it would submit duplicates that race on the same cache path.
  if squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qx "$model-$DATASET"; then
    skipped+=("$model (already queued)")
    continue
  fi

  cmd=(sbatch -J "$model-$DATASET"
       --time="$TIME_LIMIT" --mem="$MEM"
       --export="ALL,EMBEDDER=$model,DATASET=$DATASET,SPLIT=$SPLIT,SEED=0"
       slurm/extract_new_model.slurm)

  if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo "[dry-run] ${cmd[*]}"
  else
    "${cmd[@]}"
  fi
  submitted=$((submitted + 1))
done

echo
echo "submitted: $submitted job(s) for $DATASET ($SPLIT)"
if (( ${#skipped[@]} > 0 )); then
  printf 'skipped:   %s\n' "${skipped[@]}"
fi
