#!/usr/bin/env bash
# Submit SCimilarity + TranscriptFormer on immune (full transcriptome), then pooled v2 ranking.
#
#   bash scripts/submit_immune_scim_tf.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MAIN="$(sbatch --parsable slurm/immune_scimilarity_transcriptformer.slurm)"
echo "immune-scim-tf=$MAIN"
echo
echo "Monitor:"
echo "  tail -f logs/immune_scimilarity_transcriptformer_${MAIN}.out"
echo
echo "Outputs (after completion):"
echo "  runs/ranking_immune_human_v5/responses.parquet  (+ scimilarity, transcriptformer)"
echo "  runs/pooled_3dataset_v2/ability_ranking.csv"
