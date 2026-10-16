#!/bin/bash
# Train one fold on one GPU with Slurm. Defaults are for NHR@ZIB Lise (A100 nodes);
# for another cluster change the partition, qos, time and memory lines below.
#
# Usage (from kinodata/training, with slurm_out/ existing):
#   sbatch -A <account> train_fold.sbatch <fold> <split_type> <rmsd_max> <run_name> [extra train_1gpu.py args]
# Example:
#   mkdir -p slurm_out
#   sbatch -A <account> train_fold.sbatch 0 scaffold-k-fold 20 rmsd20_eps_0.05_scaffold_wl10_wh7
#
# Resubmitting the same command after a time-out continues from last.ckpt and logs to the same wandb run.
# Optional environment variables: KINODATA_ENV (conda env name or path to an unpacked conda-pack folder,
# default kinodata_env), CONDA_BASE (default ~/miniforge3),
# OUTPUT_DIR (default <repo>/runs), WANDB_ENTITY / WANDB_PROJECT / WANDB_MODE (default offline).
#
# The torch 1.12 / CUDA 11 environment supports GPUs up to sm_86 (A100 yes, H100/H200 no).

#SBATCH -p gpu-a100:shared
#SBATCH --qos=48h
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --job-name=mokddd
#SBATCH --output=slurm_out/%j_%x.out
#SBATCH --error=slurm_out/%j_%x.err

set -euo pipefail

FOLD=$1; SPLIT=$2; RMSD=$3; NAME=$4; shift 4

REPO=$(cd "$SLURM_SUBMIT_DIR/../.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/runs}

ENV=${KINODATA_ENV:-kinodata_env}
set +u   # conda activation scripts use unset variables
if [ -f "$ENV/bin/activate" ]; then
    source "$ENV/bin/activate"          # unpacked conda-pack folder
else
    source "${CONDA_BASE:-$HOME/miniforge3}/etc/profile.d/conda.sh"
    conda activate "$ENV"               # named conda env
fi
set -u

export WANDB_ENTITY=${WANDB_ENTITY:-nextaids}
export WANDB_PROJECT=${WANDB_PROJECT:-kinodata-3d_rmsd10}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-$NAME}
# Lise compute nodes have no internet: log offline, then upload from a login node with
#   wandb sync <OUTPUT_DIR>/<run_name>_fold_<fold>/wandb/offline-run-*
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTHONUNBUFFERED=1   # so prints show up in the .out file as they happen

# run inside a per-run folder so the split CSVs and wandb files of parallel jobs don't overwrite each other
RUN_DIR=$OUTPUT_DIR/${NAME}_fold_${FOLD}
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

echo "job $SLURM_JOB_ID on $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "repo $REPO @ $(git -C "$REPO" rev-parse --short HEAD), run dir $RUN_DIR"

python "$REPO/kinodata/training/train_1gpu.py" \
    --fold "$FOLD" --split_type "$SPLIT" --rmsd_max "$RMSD" \
    --csv_folder_name "$NAME" --checkpoint_name "checkpoints_$NAME" \
    --output_dir "$OUTPUT_DIR" \
    --num_workers 8 --batch_size 32 --accumulate_grad_batches 4 \
    "$@"
