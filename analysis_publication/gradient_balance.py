"""Gradient balance between the activity and the pose loss.

For one fold of the reference configuration (scaffold split, RMSD <= 10 A, eps 0.10, pose weights 10/7)
this script walks through the first training steps of an epoch, in the same batch order as training,
and computes for every step the gradient of each loss on the shared parameters (everything except the
final Linear(256, 3) output layer, whose activity and pose rows are task-specific):

- the norms of the activity and the pose gradient, overall and per block of the network;
- the cosine between them;
- the share of the combined gradient that comes from the pose loss, g_pose . g / |g|^2 (the two shares
  add up to 1), on the raw gradient and after Adam's per-parameter scaling 1 / (sqrt(v_hat) + eps)
  with v taken from the checkpoint's optimizer state.

Each quantity is computed per batch and per optimizer step (sum over `accumulate_grad_batches`
consecutive batches, as Lightning accumulates them). Models: the untrained model with the training
seed (same initial weights as the run) and every checkpoint given on the command line.

Usage (on a GPU node, from the repo root):
    python analysis_publication/gradient_balance.py --fold 0 --ckpt <best.ckpt> [<other.ckpt> ...] --out <dir>
Output: <out>/gradients_fold_<k>.csv, one row per (model, step type, step).
The data in analysis_publication/data/gradient_balance/ were computed with the trained checkpoint of each fold of the
scaffold-split pose-aware model (model labels: "untrained" = initial weights, "trained" = the trained model).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import wandb
import pytorch_lightning as pl
import kinodata.configuration as configuration
from kinodata.data.data_module import make_kinodata_module
from kinodata.model.complex_transformer import make_model
from kinodata.transform.to_complex_graph import TransformToComplexGraph

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, required=True)
parser.add_argument("--ckpt", nargs="*", default=[], help="Lightning checkpoints of the run")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--n_steps", type=int, default=400, help="training steps with both an activity and a pose batch")
parser.add_argument("--split_type", type=str, default="scaffold-k-fold")
parser.add_argument("--rmsd_max", type=float, default=10)
parser.add_argument("--activity_eps", type=float, default=0.10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--accumulate_grad_batches", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=8)
args = parser.parse_args()

# same configuration as train_1gpu.py
configuration.register(
    "sparse_transformer",
    max_num_neighbors=16,
    hidden_channels=256,
    num_attention_blocks=3,
    num_heads=8,
    act="silu",
    edge_attr_size=4,
    ln1=True,
    ln2=True,
    ln3=True,
    graph_norm=False,
    interaction_modes=["covalent", "structural"],
)
config = configuration.get("data", "training", "sparse_transformer")
config["node_types"] = ["complex"]
config["batch_size"] = args.batch_size
config["accumulate_grad_batches"] = args.accumulate_grad_batches
config["perturb_complex_positions"] = 0
config["perturb_ligand_positions"] = 0.0
config["need_distances"] = False
config["perturb_pocket_positions"] = 0.0
config["seed"] = args.seed
config["split_type"] = args.split_type
config["filter_rmsd_max_value"] = args.rmsd_max
config["k_fold"] = 5
config["split_index"] = args.fold
config["num_workers"] = args.num_workers
config["activity_eps"] = args.activity_eps
config["pose_loss_weight"] = 1.0
config["train_rmsd_max_activity"] = None
config["train_rmsd_max_pose"] = None
config["csv_save_dir"] = os.path.join(args.out, f"csv_fold_{args.fold}")  # required by RegressionModel, unused

wandb.init(mode="disabled")
pl.seed_everything(args.seed, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# same order as train_1gpu.py: data module, then model (so the untrained model has the run's initial weights)
data_module = make_kinodata_module(config, transforms=[TransformToComplexGraph(remove_heterogeneous_representation=False)])
model = make_model(config)
init_state = {k: v.clone() for k, v in model.state_dict().items()}
data_module.setup(stage="fit")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# parameter groups: shared = all but the final output layer
head = model.out[-1]
head_ids = {id(p) for p in head.parameters()}
params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
shared = [(n, p) for n, p in params if id(p) not in head_ids]


def block_of(name):
    parts = name.split(".")
    if parts[0] in ("attention_blocks", "norm_layers"):
        return f"{parts[0]}.{parts[1]}"
    if parts[0] in ("atomic_num_embedding", "lin_atom_features", "interaction_module"):
        return "input_embedding"
    return parts[0]  # aggr, out (BatchNorm + FF), bias, ...


blocks = sorted({block_of(n) for n, _ in shared})
block_index = {b: [i for i, (n, _) in enumerate(shared) if block_of(n) == b] for b in blocks}


def flat_grad(loss):
    grads = torch.autograd.grad(loss, [p for _, p in shared], allow_unused=True)
    return [torch.zeros_like(p) if g is None else g.detach() for g, (_, p) in zip(grads, shared)]


def adam_scale_from_ckpt(ckpt):
    """1 / (sqrt(v_hat) + eps) per shared parameter, from the Adam state saved in the checkpoint."""
    opt = ckpt["optimizer_states"][0]
    eps = opt["param_groups"][0].get("eps", 1e-8)
    beta2 = opt["param_groups"][0]["betas"][1]
    state = opt["state"]
    # optimizer state is indexed by the position in model.parameters()
    pos = {id(p): i for i, p in enumerate(model.parameters())}
    scales = []
    for n, p in shared:
        st = state.get(pos[id(p)])
        if st is None:  # parameter never received a gradient in training: no Adam state, and its gradient here is 0
            print(f"  no Adam state for {n}", flush=True)
            scales.append(torch.ones_like(p))
            continue
        step = float(st["step"])
        v_hat = st["exp_avg_sq"].to(device) / (1 - beta2 ** step)
        scales.append(1.0 / (v_hat.sqrt() + eps))
    return scales


def stats(g_act, g_pose, scale):
    """Norms, cosine and pose share of the combined gradient (shared parameters)."""
    out = {}
    a = torch.cat([g.flatten() for g in g_act])
    p = torch.cat([g.flatten() for g in g_pose])
    out["norm_act"] = a.norm().item()
    out["norm_pose"] = p.norm().item()
    out["cos"] = (torch.dot(a, p) / (a.norm() * p.norm()).clamp_min(1e-30)).item()
    tot = a + p
    out["share_pose"] = (torch.dot(p, tot) / torch.dot(tot, tot).clamp_min(1e-30)).item()
    if scale is not None:
        s = torch.cat([x.flatten() for x in scale])
        a_s, p_s = a * s, p * s
        tot_s = a_s + p_s
        out["norm_act_adam"] = a_s.norm().item()
        out["norm_pose_adam"] = p_s.norm().item()
        out["share_pose_adam"] = (torch.dot(p_s, tot_s) / torch.dot(tot_s, tot_s).clamp_min(1e-30)).item()
    for b, idx in block_index.items():
        out[f"norm_act__{b}"] = torch.cat([g_act[i].flatten() for i in idx]).norm().item()
        out[f"norm_pose__{b}"] = torch.cat([g_pose[i].flatten() for i in idx]).norm().item()
    return out


models = [("init", None)] + [(Path(c).name, c) for c in args.ckpt]
rows = []
for label, ckpt_path in models:
    if ckpt_path is None:
        model.load_state_dict(init_state)
        epoch, scale = 0, None
    else:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        epoch = ckpt["epoch"]
        scale = adam_scale_from_ckpt(ckpt)
    model.to(device)
    model.train()  # BatchNorm on batch statistics, as in training (dropout is 0)
    print(f"[fold {args.fold}] model {label}: epoch {epoch}", flush=True)

    # same batches for every model: re-seed before building the iterators (activity first, as CombinedLoader)
    pl.seed_everything(args.seed, workers=True)
    it_act = iter(data_module.dm_a.train_dataloader())
    it_pose = iter(data_module.dm_b.train_dataloader())
    acc_act = acc_pose = None
    for step in range(args.n_steps):
        batch_act = next(it_act).to(device)
        batch_pose = next(it_pose).to(device)
        loss_act = model.compute_loss_activity(model(batch_act), batch_act)
        loss_pose = model.compute_loss_pose(model(batch_pose), batch_pose)
        g_act, g_pose = flat_grad(loss_act), flat_grad(loss_pose)
        rows.append(dict(fold=args.fold, model=label, epoch=epoch, kind="batch", step=step,
                         loss_act=loss_act.item(), loss_pose=loss_pose.item(), **stats(g_act, g_pose, scale)))
        acc_act = g_act if acc_act is None else [x + y for x, y in zip(acc_act, g_act)]
        acc_pose = g_pose if acc_pose is None else [x + y for x, y in zip(acc_pose, g_pose)]
        if (step + 1) % args.accumulate_grad_batches == 0:
            rows.append(dict(fold=args.fold, model=label, epoch=epoch, kind="optimizer_step",
                             step=step // args.accumulate_grad_batches, **stats(acc_act, acc_pose, scale)))
            acc_act = acc_pose = None
        if step % 100 == 0:
            print(f"  step {step}: |g_act| {rows[-1]['norm_act']:.3g}  |g_pose| {rows[-1]['norm_pose']:.3g}", flush=True)

df = pd.DataFrame(rows)
os.makedirs(args.out, exist_ok=True)
df.to_csv(os.path.join(args.out, f"gradients_fold_{args.fold}.csv"), index=False)

summary = (df.assign(ratio=df.norm_pose / df.norm_act)
             .groupby(["model", "epoch", "kind"])[["ratio", "cos", "share_pose"] + (["share_pose_adam"] if "share_pose_adam" in df else [])]
             .median())
print(summary.to_string())
