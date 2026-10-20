"""Fill the wandb run summary from the run history.

Runs logged offline with the old wandb 0.13 of the torch-1.12 env and uploaded with a recent
client (`wandb sync` / `wandb beta sync --live`) have an empty summary: the offline client never
records the logged metrics in it. The values are in the history, so this script writes the last
value of every metric (min / max for the metrics RegressionModel.define_metrics declares) into
the summary, which is what the wandb runs table shows.

Needs a recent wandb (not the one of the training env), e.g. on iris:
    PYTHONPATH=runs/tools/wandb_client ~/miniforge3/bin/python kinodata/training/wandb_fill_summary.py \
        nextaids/kinodata-3d_rmsd10/<run_id> [more run paths]
"""
import math
import sys

import wandb

# as in RegressionModel.define_metrics
SUMMARY_AGG = {"val/mae_activity": "min", "val/corr_activity": "max"}


def fill_summary(api, path):
    run = api.run(path)
    last, best = {}, {}
    for row in run.scan_history():
        for key, value in row.items():
            if key.startswith("_") or not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            last[key] = value
            if key in SUMMARY_AGG:
                op = min if SUMMARY_AGG[key] == "min" else max
                best[key] = op(best.get(key, value), value)
    summary = dict(last)
    for key, value in best.items():
        summary[key] = {SUMMARY_AGG[key]: value}
    run.summary.update(summary)
    run.update()
    print(f"{path}: {len(summary)} summary values written")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    api = wandb.Api(timeout=60)
    for path in sys.argv[1:]:
        fill_summary(api, path)
