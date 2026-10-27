#!/bin/bash
#SBATCH --job-name=tx1-sif
#SBATCH --account=p32593
#SBATCH --partition=short
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Tahoe-x1's flash-attn wheels need glibc >= 2.32 and Quest runs RHEL 8 (glibc 2.28), so the model runs
# from the upstream Docker image (PyTorch, CUDA and flash-attn preinstalled) through Apptainer.
module load apptainer/1.1.9
cd /gpfs/projects/p32593/scfm-irt/cache/tahoex1_workdir
export APPTAINER_CACHEDIR=$PWD/apptainer_cache APPTAINER_TMPDIR=$PWD/apptainer_tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
apptainer pull --force sif/tahoe-x1_1.0.0.sif docker://ghcr.io/tahoebio/tahoe-x1:1.0.0
ls -la sif/
apptainer exec sif/tahoe-x1_1.0.0.sif python -c "import torch, flash_attn; print('torch', torch.__version__, 'flash', flash_attn.__version__)"
