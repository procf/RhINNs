#!/bin/bash
#SBATCH --job-name=pgdiff-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=pgdiff-train-%j.out
#SBATCH --error=pgdiff-train-%j.err

# Train the unconditional EDM model on the joint (initial, final) velocity fields.
# Usage:  sbatch slurm/train.sbatch cylinder      (or: oldroyd)
# Edit the partition, modules and environment name for your cluster.

set -euo pipefail
CASE="${1:-cylinder}"

module load anaconda3 || true
source activate pgdiff-flows

# Run from the repository root so dnnlib/ and torch_utils/ are importable.
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

case "$CASE" in
  cylinder)
    torchrun --standalone --nproc_per_node=1 train.py \
      --outdir=training-runs/cylinder --data=data/NS-merged \
      --cond=0 --arch=ddpmpp --batch=15 --tick=10 --snap=25 --dump=100 \
      --duration=300 --ema=0.05
    ;;
  oldroyd)
    torchrun --standalone --nproc_per_node=1 train.py \
      --outdir=training-runs/oldroyd --data=data/Oldroyd-merged \
      --cond=0 --arch=ddpmpp --batch=64 --tick=10 --snap=10 --dump=100 \
      --duration=20 --ema=0.05
    ;;
  *)
    echo "Unknown case: $CASE (use cylinder or oldroyd)"; exit 1 ;;
esac
