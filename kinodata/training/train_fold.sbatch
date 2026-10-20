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
# Example (iris; WANDB_SYNC_PY = python with a recent wandb for live upload, see CLAUDE.md):
#   mkdir -p slurm_out; export KINODATA_ENV=kinodata_backup WANDB_SYNC_PY=<repo>/runs/tools/wandb_python
#   sbatch -p gpu --gres=gpu:l40s:1 --time=3-00:00:00 train_fold.sbatch 0 scaffold-k-fold 10 rmsd10_eps0.10_scaffold_wl10_wh7_repro
#
# Resubmitting the same command after a time-out continues from last.ckpt and logs to the same wandb run,
# so a new run needs a new run name.
# Optional environment variables: KINODATA_ENV (conda env name or path to an unpacked conda-pack folder,
# default kinodata_env), CONDA_BASE (default ~/miniforge3), OUTPUT_DIR (default <repo>/runs),
# WANDB_ENTITY / WANDB_PROJECT / WANDB_MODE (default offline).
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
# offline by default: Lise nodes have no internet, and the wandb 0.13 of the torch-1.12 env can no
# longer create runs on the wandb server (fails in server_settings_introspection). Upload afterwards
# from a login node with a recent wandb client:
#   wandb sync --entity $WANDB_ENTITY --project $WANDB_PROJECT <OUTPUT_DIR>/<run_name>_fold_<fold>/wandb/offline-run-*
# or set WANDB_SYNC_PY to a python with a recent wandb: the job then uploads the run live while it trains
# (needs internet on the node) and fills the run summary at the end (wandb_fill_summary.py).
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTHONUNBUFFERED=1   # so prints show up in the .out file as they happen

# run inside a per-run folder so the split CSVs and wandb files of parallel jobs don't overwrite each other
RUN_DIR=$OUTPUT_DIR/${NAME}_fold_${FOLD}
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

echo "job $SLURM_JOB_ID on $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader), wandb $WANDB_MODE"
echo "repo $REPO @ $(git -C "$REPO" rev-parse --short HEAD), run dir $RUN_DIR"

# optional live upload of the offline run (log: wandb_live_sync.log in the run dir)
SYNC_PID=
if [ "$WANDB_MODE" = offline ] && [ -n "${WANDB_SYNC_PY:-}" ]; then
    START=$(date +%s)
    (
        set +e +o pipefail   # the wandb folder does not exist at first; keep polling
        # wait for the offline-run folder that this job's wandb.init creates
        for _ in $(seq 1 360); do
            D=$(ls -dt wandb/offline-run-* 2>/dev/null | head -1)
            [ -n "$D" ] && [ "$(stat -c %Y "$D")" -ge "$START" ] && break
            D=; sleep 10
        done
        [ -n "$D" ] && exec "$WANDB_SYNC_PY" -m wandb beta sync --live --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" "$D"
    ) > wandb_live_sync.log 2>&1 &
    SYNC_PID=$!
fi

STATUS=0
python "$REPO/kinodata/training/train_1gpu.py" \
    --fold "$FOLD" --split_type "$SPLIT" --rmsd_max "$RMSD" \
    --csv_folder_name "$NAME" --checkpoint_name "checkpoints_$NAME" \
    --output_dir "$OUTPUT_DIR" \
    --num_workers 8 --batch_size 32 --accumulate_grad_batches 4 \
    "$@" || STATUS=$?

if [ -n "$SYNC_PID" ]; then
    # give the live sync up to 5 min to upload the end of the run, then stop it
    for _ in $(seq 1 30); do kill -0 "$SYNC_PID" 2>/dev/null || break; sleep 10; done
    kill "$SYNC_PID" 2>/dev/null || true
    # the old offline client leaves the summary empty; rebuild it from the history
    RUN_ID=$(ls -dt wandb/offline-run-* 2>/dev/null | head -1 | sed 's/.*-//' || true)
    [ -n "$RUN_ID" ] && "$WANDB_SYNC_PY" "$REPO/kinodata/training/wandb_fill_summary.py" \
        "$WANDB_ENTITY/$WANDB_PROJECT/$RUN_ID" >> wandb_live_sync.log 2>&1 || true
fi
exit $STATUS
