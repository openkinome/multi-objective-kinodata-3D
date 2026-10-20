#!/bin/bash
# Train one fold on one GPU with Slurm. The file holds only machine-independent settings;
# partition, GPU type, account and qos go on the sbatch command line.
#
# Usage (from kinodata/training, with slurm_out/ existing):
#   sbatch <machine flags> train_fold.sbatch <fold> <split_type> <rmsd_max> <run_name> [extra train_1gpu.py args]
#
# Machine flags:
#   iris (MSKCC):  sbatch -p gpu --gres=gpu:l40s:1 --time=3-00:00:00 ...   (with KINODATA_ENV=kinodata_backup exported)
#   Lise (NHR@ZIB): sbatch -A beb00059 -p gpu-a100:shared --qos=48h ...    (Lise does not pass the shell environment:
#                   add --export=ALL,KINODATA_ENV=... if needed)
# Example (iris):
#   mkdir -p slurm_out; export KINODATA_ENV=kinodata_backup
#   sbatch -p gpu --gres=gpu:l40s:1 --time=3-00:00:00 train_fold.sbatch 0 scaffold-k-fold 20 rmsd20_scaffold_wl10_wh7_repro
#
# Resubmitting the same command after a time-out continues from last.ckpt and logs to the same wandb run,
# so a new run needs a new run name.
# Optional environment variables: KINODATA_ENV (conda env name or path to an unpacked conda-pack folder,
# default kinodata_env), CONDA_BASE (default ~/miniforge3), OUTPUT_DIR (default <repo>/runs),
# WANDB_ENTITY / WANDB_PROJECT / WANDB_MODE (default: online if the node reaches wandb, else offline).
#
# The torch 1.12 / CUDA 11 environment supports GPUs up to sm_86 (L40S, A100, A40 yes; H100/H200 no).

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
# nodes without internet (Lise) log offline; upload later from a login node with
#   wandb sync <OUTPUT_DIR>/<run_name>_fold_<fold>/wandb/offline-run-*
if [ -z "${WANDB_MODE:-}" ]; then
    if timeout 10 curl -sI https://api.wandb.ai >/dev/null 2>&1; then WANDB_MODE=online; else WANDB_MODE=offline; fi
fi
export WANDB_MODE
export PYTHONUNBUFFERED=1   # so prints show up in the .out file as they happen

# run inside a per-run folder so the split CSVs and wandb files of parallel jobs don't overwrite each other
RUN_DIR=$OUTPUT_DIR/${NAME}_fold_${FOLD}
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

echo "job $SLURM_JOB_ID on $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader), wandb $WANDB_MODE"
echo "repo $REPO @ $(git -C "$REPO" rev-parse --short HEAD), run dir $RUN_DIR"

python "$REPO/kinodata/training/train_1gpu.py" \
    --fold "$FOLD" --split_type "$SPLIT" --rmsd_max "$RMSD" \
    --csv_folder_name "$NAME" --checkpoint_name "checkpoints_$NAME" \
    --output_dir "$OUTPUT_DIR" \
    --num_workers 8 --batch_size 32 --accumulate_grad_batches 4 \
    "$@"
