#!/bin/bash
#SBATCH --job-name=pgdiff-gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=pgdiff-gen-%j.out
#SBATCH --error=pgdiff-gen-%j.err

# Physics-guided sampling for one test case.
# Usage:  sbatch slurm/generate.sbatch <cylinder|oldroyd> [extra options for the generator]
# Examples:
#   sbatch slurm/generate.sbatch cylinder --datapath data/NS-merged-test/u0_0p32137.npy --zeta-obs-a 100 --seed 1
#   sbatch slurm/generate.sbatch oldroyd  --datapath data/Oldroyd-merged-test/Wi_0p9993.npy --seed 2
# Edit the partition, modules and environment name for your cluster.

set -euo pipefail
CASE="${1:-cylinder}"
shift || true

module load anaconda3 || true
source activate pgdiff-flows

# Run from the repository root so dnnlib/ and torch_utils/ are importable and the
# relative paths in the configs resolve.
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "Host: $(hostname)"
nvidia-smi || true
python -m "scripts.generate_${CASE}" --config "configs/${CASE}-forward.yaml" "$@"
