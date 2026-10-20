# data_module.py  — unified for random / scaffold (joint) / pocket (joint or independent)

from __future__ import annotations
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict
from collections import Counter

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from numpy.random import default_rng
from sklearn.model_selection import GroupKFold
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric.data.lightning_datamodule import LightningDataset
from torch_geometric.transforms import Compose
from pytorch_lightning.utilities.combined_loader import CombinedLoader

from kinodata.configuration import Config
from kinodata.data.data_split import Split
from kinodata.data.dataset import KinodataDocked, Filtered
from kinodata.data.dataset_davids_data import DavidsdataDocked, Filtered_david
from kinodata.types import NodeType
import kinodata.transform as T

# from your consolidated grouped_split.py
from kinodata.data.grouped_split import (
    KinodataKFoldSplit,
    joint_pocket_kfold_split,    # synchronized pocket folds (activity + pose)
    cap_split_by_max_share,      # optional per-pocket cap within a split
)

Kwargs = Dict[str, Any]


# --- Scaffold diagnostics  ---


def _scaffold_array(ds):
    return np.array([ds[i].scaffold for i in range(len(ds))], dtype=object)

def _summarize_scaffold_split(scaffolds: np.ndarray, split, tag: str):
    def nuniq(ix): return len(set(scaffolds[i] for i in ix))
    def largest(ix):
        c = Counter(scaffolds[i] for i in ix)
        return c.most_common(1)[0][1] if c else 0
    def nrows(ix): return len(ix) if ix is not None else 0
    print(f"[joint-scaffold] {tag}")
    print(f"  train: {nuniq(split.train_split):5d} scaffolds, {nrows(split.train_split):5d} mols (largest={largest(split.train_split)})")
    print(f"  val  : {nuniq(split.val_split):5d} scaffolds, {nrows(split.val_split):5d} mols (largest={largest(split.val_split)})")
    print(f"  test : {nuniq(split.test_split):5d} scaffolds, {nrows(split.test_split):5d} mols (largest={largest(split.test_split)})")

def _assert_no_leakage(scaffolds: np.ndarray, split, tag: str):
    S_tr = set(scaffolds[i] for i in (split.train_split or []))
    S_va = set(scaffolds[i] for i in (split.val_split or []))
    S_te = set(scaffolds[i] for i in (split.test_split or []))
    leak_tv, leak_tt = S_tr & S_va, S_tr & S_te
    if leak_tv or leak_tt:
        raise RuntimeError(
            f"[leakage:{tag}] found scaffolds in train ∩ val: {len(leak_tv)} "
            f"and train ∩ test: {len(leak_tt)}"
        )
    print(f"[no-leak:{tag}] |train|={len(S_tr)} |val|={len(S_va)} |test|={len(S_te)} "
          f"overlap(train,val)={len(leak_tv)} overlap(train,test)={len(leak_tt)}")

def _pose_activity_coverage(scaff_act, scaff_pose, split_act, split_pose):
    def cov(p_idx, a_idx, name):
        P = set(scaff_pose[i] for i in p_idx)
        A = set(scaff_act[i]  for i in a_idx)
        inter = P & A
        denom = len(P) if P else 1
        pct = 100.0 * len(inter) / denom
        print(f"  [{name}] pose scaffolds: {len(P)}, overlap with activity: {len(inter)} ({pct:.1f}%)")
        # print 5 examples like your logs
        if inter:
            ex = list(inter)[:5]
            print(f"           examples: {ex}")
    print("[joint-scaffold] pose→activity scaffold coverage:")
    cov(split_pose.train_split, split_act.train_split, "train")
    cov(split_pose.val_split,   split_act.val_split,   "val")
    cov(split_pose.test_split,  split_act.test_split,  "test")

def _dump_smiles_counts(ds, split, label: str):
    smiles = [d.smiles for d in ds]
    n_tr = len([smiles[i] for i in split.train_split])
    n_va = len([smiles[i] for i in split.val_split])
    n_te = len([smiles[i] for i in split.test_split])
    print(f"the len of the train {label} smiles is {n_tr}")
    scaff = [d.scaffold for d in ds]
    n_tr_scaff = len([scaff[i] for i in split.train_split])
    print(f"the len of the train {label} scaffold is {n_tr_scaff}")


def compose(transforms: Optional[list]) -> Optional[Callable]:
    return None if transforms is None else Compose(transforms)


#random split checks

def _set_scaffolds(ds):
    # unique scaffold set
    return set(d.scaffold for d in ds)

def _idx_scaffolds(ds, idxs):
    return set(ds[i].scaffold for i in (idxs or []))

def _jaccard(A, B):
    U = A | B
    I = A & B
    return len(U), len(I), (len(I) / len(U) if U else 0.0)

def _print_overall_overlap(activity_ds, pose_ds):
    A = _set_scaffolds(activity_ds)
    P = _set_scaffolds(pose_ds)
    U, I, J = _jaccard(A, P)
    print("=== Overall Scaffold Overlap ===")
    print(f"Activity scaffolds: {len(A)}")
    print(f"Pose scaffolds:     {len(P)}")
    print(f"Union:              {U}")
    print(f"Intersection:       {I}")
    print(f"Jaccard:            {J:.4f}")

def _print_fold_overlap(activity_ds, pose_ds, split_act, split_pose):
    print("=== Scaffold overlap report (current independent splits) ===")
    # GLOBAL: use the full datasets (independent of folds)
    A_all = _set_scaffolds(activity_ds)
    P_all = _set_scaffolds(pose_ds)
    U,I,J = _jaccard(A_all, P_all)
    print(f"[GLOBAL]  act={len(A_all):5d}  pose={len(P_all):5d}  ∪={U:5d}  ∩={I:5d}  Jaccard={J:.3f}")

    # Per split
    A_tr = _idx_scaffolds(activity_ds, split_act.train_split)
    A_va = _idx_scaffolds(activity_ds, split_act.val_split)
    A_te = _idx_scaffolds(activity_ds, split_act.test_split)

    P_tr = _idx_scaffolds(pose_ds, split_pose.train_split)
    P_va = _idx_scaffolds(pose_ds, split_pose.val_split)
    P_te = _idx_scaffolds(pose_ds, split_pose.test_split)

    for tag, A_set, P_set in [
        ("TRAIN", A_tr, P_tr),
        ("VAL",   A_va, P_va),
        ("TEST",  A_te, P_te),
    ]:
        U,I,J = _jaccard(A_set, P_set)
        print(f"[{tag}]  act={len(A_set):5d}  pose={len(P_set):5d}  ∪={U:5d}  ∩={I:5d}  Jaccard={J:.3f}")



def load_precomputed_split(path_or_cfg) -> Split:
    # supports config.data_split being a string path or a config with .data_split
    p = Path(path_or_cfg if isinstance(path_or_cfg, (str, Path)) else path_or_cfg.data_split)
    if not p.exists():
        raise FileNotFoundError(str(p))
    print(f"Loading split from {p}..")
    split = Split.from_data_frame(pd.read_csv(p))
    split.source_file = str(p)
    return split

def assert_unique_value(key: str, *kwarg_dicts: Optional[Kwargs], msg: str = ""):
    values = []
    for kwarg_dict in kwarg_dicts:
        if not kwarg_dict:
            continue
        if key in kwarg_dict:
            values.append(kwarg_dict[key])
    assert len(set(values)) <= 1, msg

class CombinedDataModule(pl.LightningDataModule):
    def __init__(self, dm_a: LightningDataset, dm_b: LightningDataset):
        super().__init__()
        self.dm_a, self.dm_b = dm_a, dm_b

    def setup(self, stage=None):
        self.dm_a.setup(stage=stage)
        self.dm_b.setup(stage=stage)

    def train_dataloader(self):
        return CombinedLoader(
            {"activity": self.dm_a.train_dataloader(),
             "pose":     self.dm_b.train_dataloader()},
            mode="max_size"
        )

    def val_dataloader(self):
        return CombinedLoader(
            {"activity": self.dm_a.val_dataloader(),
             "pose":     self.dm_b.val_dataloader()},
            mode="sequential"
        )

    def test_dataloader(self):
        return CombinedLoader(
            {"activity": self.dm_a.test_dataloader(),
             "pose":     self.dm_b.test_dataloader()},
            mode="sequential"
        )

def make_data_module(
    split: Split,
    batch_size: int,
    num_workers: int,
    *,
    dataset_cls: Optional[type[InMemoryDataset]] = None,
    dataset_instance: Optional[InMemoryDataset] = None,
    train_kwargs: Kwargs | None = None,
    val_kwargs: Optional[Kwargs] = None,
    test_kwargs: Optional[Kwargs] = None,
    one_time_transform: Optional[Callable[[InMemoryDataset], InMemoryDataset]] = None,
    normalization: bool = False,
    normalization_idx: Optional[List[int]] = None,
    **kwargs,
) -> LightningDataset:
    """normalization_idx: indices of full_dataset whose targets set the normalization mean/std
    (default: the training split). Used when the training split is filtered but the targets must
    stay on the scale of the unfiltered training split."""
    assert (dataset_cls is None) ^ (dataset_instance is None)

    def subset(full_ds: InMemoryDataset, idx: List[int]):
        view = full_ds[idx]
        return one_time_transform(view) if one_time_transform else view

    full_dataset = dataset_instance if dataset_instance is not None else dataset_cls(**(train_kwargs or {}))

    train_dataset = subset(full_dataset, split.train_split)
    val_dataset   = subset(full_dataset, split.val_split) if split.val_split else None
    test_dataset  = subset(full_dataset, split.test_split) if split.test_split else None

    # attach transforms AFTER subset
    train_dataset.transform = (train_kwargs or {}).get("transform")
    if val_dataset is not None:
        val_dataset.transform   = (val_kwargs   or {}).get("transform")
    if test_dataset is not None:
        test_dataset.transform  = (test_kwargs  or {}).get("transform")

    # optional target normalization (activity)
    if normalization and train_dataset is not None:
        norm_dataset = train_dataset if normalization_idx is None else full_dataset[normalization_idx]
        y = torch.stack([d.y for d in norm_dataset])
        mu, sigma = y.mean().item(), y.std().item()
        print(f"[normalize before y] mean={mu:.4f} std={sigma:.4f}")

        class ScaleY:
            def __init__(self, mu, sigma): 
                self.mu, self.sigma = mu, sigma
            def __call__(self, data): 
                data.y = (data.y - self.mu) / self.sigma
                return data

        def _add(tf_new, tf_old): 
            return Compose([tf_new, tf_old]) if tf_old is not None else tf_new
        
        norm_tf = ScaleY(mu, sigma)
        
        train_dataset.transform = _add(norm_tf, train_dataset.transform)
        val_dataset.transform   = _add(norm_tf, val_dataset.transform)
        test_dataset.transform  = _add(norm_tf, test_dataset.transform)
        
        # --- AFTER normalization ---
        normalized_values = torch.tensor([d.y.item() for d in train_dataset])
        print("**After Normalization**")
        print(f"  Mean: {normalized_values.mean().item():.4f}, Std: {normalized_values.std().item():.4f}")
        print(f"  Sample values: {normalized_values[:5].squeeze().tolist()}")
        print("===========================\n")

    return LightningDataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4,
        **kwargs,
    )

def _filter_trainval_by_rmsd(ds, split: Split, max_rmsd: Optional[float], tag: str) -> Split:
    """Keep only training/validation samples with predicted_rmsd <= max_rmsd; the test split is unchanged."""
    if max_rmsd is None:
        return split
    store = ds._data if getattr(ds, "_data", None) is not None else ds.data
    rmsd = store.predicted_rmsd.view(-1).numpy()
    assert len(rmsd) == len(ds), f"{tag}: one predicted_rmsd per sample expected"

    def keep(idx):
        idx = np.asarray(idx, dtype=int)
        return idx[rmsd[idx] <= max_rmsd].tolist()

    filtered = Split(keep(split.train_split), keep(split.val_split), split.test_split)
    print(f"[{tag} train/val RMSD <= {max_rmsd}] train {split.train_size} -> {filtered.train_size}, "
          f"val {split.val_size} -> {filtered.val_size}, test {split.test_size} (unchanged)")
    return filtered


# ------------------------------------------
# Joint SCAFFOLD split builder (aligned folds)
# ------------------------------------------

def make_joint_scaffold_splits(
    activity_ds,
    pose_ds,
    *,
    k: int,
    split_index: int = 0,
    val_frac: float = 0.5,
    seed: int = 0,
    max_samples_per_scaffold_activity: int | None = None,
    max_samples_per_scaffold_pose: int | None = None,
) -> Tuple[Split, Split]:
    """One outer GroupKFold by scaffold across BOTH datasets; map back per dataset; inner val/test by scaffold."""
    scaff_act = np.array([activity_ds[i].scaffold for i in range(len(activity_ds))], dtype=object)
    scaff_pose = np.array([pose_ds[i].scaffold     for i in range(len(pose_ds))],     dtype=object)
    idx_act, idx_pose = np.arange(len(activity_ds)), np.arange(len(pose_ds))

    def cap_per_scaffold(scaff, idx, cap):
        if cap is None: return idx
        bucket = defaultdict(list)
        for i, s in zip(idx, scaff[idx]): bucket[s].append(i)
        keep = []
        for s, ids in bucket.items(): keep.extend(ids[:cap])
        return np.array(keep, dtype=int)

    keep_act  = cap_per_scaffold(scaff_act,  idx_act,  max_samples_per_scaffold_activity)
    keep_pose = cap_per_scaffold(scaff_pose, idx_pose, max_samples_per_scaffold_pose)

    joint_scaffolds = np.concatenate([scaff_act[keep_act], scaff_pose[keep_pose]])
    src_flags = np.concatenate([np.zeros_like(keep_act, dtype=np.int8),
                                np.ones_like(keep_pose, dtype=np.int8)])
    src_local = np.concatenate([keep_act, keep_pose])

    gkf = GroupKFold(n_splits=k)
    X = np.zeros((len(joint_scaffolds), 1))
    folds = list(gkf.split(X, groups=joint_scaffolds))
    assert 0 <= split_index < k
    tr_all, ho_all = folds[split_index]

    rng = default_rng(seed)
    heldout_scafs = np.unique(joint_scaffolds[ho_all])
    perm = rng.permutation(len(heldout_scafs))
    pivot = int(len(heldout_scafs) * val_frac)
    val_scafs  = set(heldout_scafs[perm[:pivot]])
    test_scafs = set(heldout_scafs[perm[pivot:]])
    train_scafs = set(np.unique(joint_scaffolds[tr_all]))

    def select_from_kept(scaff_all, kept_idx, which_scafs, cap=None):
        if len(kept_idx) == 0: return kept_idx
        mask = np.array([s in which_scafs for s in scaff_all[kept_idx]], dtype=bool)
        sel = kept_idx[mask]
        if cap is None: return sel
        buckets = defaultdict(list)
        for i in sel: buckets[scaff_all[i]].append(i)
        out = []
        for s, ids in buckets.items(): out.extend(ids[:cap])
        return np.array(out, dtype=int)

    act_tr = select_from_kept(scaff_act,  keep_act,  train_scafs, max_samples_per_scaffold_activity)
    act_va = select_from_kept(scaff_act,  keep_act,  val_scafs,   max_samples_per_scaffold_activity)
    act_te = select_from_kept(scaff_act,  keep_act,  test_scafs,  max_samples_per_scaffold_activity)

    pose_tr = select_from_kept(scaff_pose, keep_pose, train_scafs, max_samples_per_scaffold_pose)
    pose_va = select_from_kept(scaff_pose, keep_pose, val_scafs,   max_samples_per_scaffold_pose)
    pose_te = select_from_kept(scaff_pose, keep_pose, test_scafs,  max_samples_per_scaffold_pose)

    return Split(act_tr, act_va, act_te), Split(pose_tr, pose_va, pose_te)

# -------------------------
# Main builder (unified API)
# -------------------------

def make_kinodata_module(
    config: Config,
    transforms=None,
    one_time_transform=None,
) -> LightningDataset:
    # datasets (optional RMSD filtering)
    poseBaseCls     = partial(DavidsdataDocked, remove_hydrogen=config.remove_hydrogen)
    activityBaseCls = partial(KinodataDocked,   remove_hydrogen=config.remove_hydrogen)

    if getattr(config, "filter_rmsd_max_value", None) is not None:
        # the unfiltered sources are only read if the filtered cache has to be built
        poseCls     = Filtered_david(poseBaseCls(load=False),     T.FilterDockingRMSD(config.filter_rmsd_max_value))
        activityCls = Filtered(     activityBaseCls(load=False),  T.FilterDockingRMSD(config.filter_rmsd_max_value))
    else:
        poseCls, activityCls = poseBaseCls, activityBaseCls

    pose_ds, activity_ds = poseCls(), activityCls()

    # transforms / augmentations
    transforms = [] if transforms is None else transforms
    augmentations = []
    if getattr(config, "perturb_ligand_positions", 0.0) and config.need_distances:
        augmentations.append(T.PerturbAtomPositions(NodeType.Ligand, std=config.perturb_ligand_positions))
    if getattr(config, "perturb_pocket_positions", 0.0) and config.need_distances:
        augmentations.append(T.PerturbAtomPositions(NodeType.Pocket, std=config.perturb_pocket_positions))
    if getattr(config, "perturb_complex_positions", 0.0) > 0.0:
        augmentations.append(T.PerturbAtomPositions(NodeType.Complex, std=config.perturb_complex_positions))

    train_transform = compose(augmentations + transforms)
    val_transform   = compose(transforms)
    print(f"[TF] train: {train_transform} | val: {val_transform}")

    # -----------------
    # compute the splits
    # -----------------

    # prefer precomputed when provided
    if getattr(config, "data_split", None):
        sp = load_precomputed_split(config)
        split_act  = sp.remap_index(activity_ds.ident_index_map())
        split_pose = sp.remap_index(pose_ds.ident_index_map())
    else:
        stype   = config.split_type                     # "random-k-fold", "scaffold-k-fold", "pocket-k-fold"
        k       = config.k_fold
        sidx    = config.split_index
        vfrac   = getattr(config, "val_frac", 0.5)
        seed    = getattr(config, "split_seed", 0)
        # caps (optional)
        cap_act = getattr(config, "cap_scaffold_activity", None)
        cap_pose= getattr(config, "cap_scaffold_pose",     None)

        if stype == "random-k-fold":
            # independent random splits per dataset (no joint logic)
            get_split = lambda raw: KinodataKFoldSplit("random-k-fold", k, val_frac=vfrac, seed=seed).split(raw)[sidx]
            split_act  = get_split(activity_ds)
            split_pose = get_split(pose_ds)

            _print_overall_overlap(activity_ds, pose_ds)
            _print_fold_overlap(activity_ds, pose_ds, split_act, split_pose)

            print(f"Split kinodata: Train size {split_act.train_size}, Val size {split_act.val_size}, Test size {split_act.test_size}")
            print(f"Split kinodocked: Train size {split_pose.train_size}, Val size {split_pose.val_size}, Test size {split_pose.test_size}")


        elif stype == "scaffold-k-fold":
            # joint scaffold split (aligned activity/pose)
            split_act, split_pose = make_joint_scaffold_splits(
                activity_ds, pose_ds,
                k=k, split_index=sidx, val_frac=vfrac, seed=seed,
                max_samples_per_scaffold_activity=cap_act,
                #max_samples_per_scaffold_pose=cap_pose,
                max_samples_per_scaffold_pose=3000,
            )

            # ------------------------
            # diagnostics dump
            # ------------------------
            # --- Optional: print scaffold diagnostics for scaffold-k-fold
            scaff_act = _scaffold_array(activity_ds)
            scaff_pose = _scaffold_array(pose_ds)

            _summarize_scaffold_split(scaff_act,  split_act,  "activity")
            _summarize_scaffold_split(scaff_pose, split_pose, "pose")
            _pose_activity_coverage(scaff_act, scaff_pose, split_act, split_pose)

            _assert_no_leakage(scaff_act,  split_act,  "activity")
            _assert_no_leakage(scaff_pose, split_pose, "pose")

            _dump_smiles_counts(activity_ds, split_act,  "act")
            _dump_smiles_counts(pose_ds,     split_pose, "pose")

            print(f"Split kinodata: Train size {split_act.train_size}, Val size {split_act.val_size}, Test size {split_act.test_size}")
            print(f"Split kinodocked: Train size {split_pose.train_size}, Val size {split_pose.val_size}, Test size {split_pose.test_size}")

        elif stype == "pocket-k-fold":
            # joint pocket split (aligned) by default; allow opt-out
            if getattr(config, "sync_pocket_splits", True):
                dual_folds, act_groups, pose_groups = joint_pocket_kfold_split(
                    activity_ds, pose_ds, k=k, val_frac=vfrac, seed=seed,
                    enforce_no_leak=True
                )
                dual = dual_folds[sidx]
                split_act, split_pose = dual.activity, dual.pose

                # Optional per-pocket capping WITHIN splits (share-based)
                if any(getattr(config, k, None) is not None for k in
                       ("cap_share_activity","cap_min_activity","cap_max_activity",
                        "cap_share_pose","cap_min_pose","cap_max_pose")):
                    split_act = cap_split_by_max_share(
                        split_act, act_groups,
                        max_share=getattr(config, "cap_share_activity", 0.10),
                        min_cap =getattr(config, "cap_min_activity",   1),
                        max_cap =getattr(config, "cap_max_activity",   None),
                        seed=seed,
                    )
                    split_pose = cap_split_by_max_share(
                        split_pose, pose_groups,
                        max_share=getattr(config, "cap_share_pose", 0.10),
                        min_cap =getattr(config, "cap_min_pose",   1),
                        max_cap =getattr(config, "cap_max_pose",   None),
                        seed=seed,
                    )
            else:
                get_split = lambda raw: KinodataKFoldSplit("pocket-k-fold", k, val_frac=vfrac, seed=seed).split(raw)[sidx]
                split_act  = get_split(activity_ds)
                split_pose = get_split(pose_ds)
        else:
            raise ValueError(f"Unknown split_type={stype}")




    # optional RMSD filter on the training and validation parts only (ablations: train on a
    # pose-quality range, evaluate on the unchanged test set). Activity targets keep the
    # normalization of the unfiltered training split, so test values stay comparable.
    act_norm_idx = split_act.train_split
    split_act = _filter_trainval_by_rmsd(activity_ds, split_act, getattr(config, "train_rmsd_max_activity", None), "activity")
    split_pose = _filter_trainval_by_rmsd(pose_ds, split_pose, getattr(config, "train_rmsd_max_pose", None), "pose")

    def dump_smiles_scaffolds(ds, split, prefix):
        smiles = [d.smiles for d in ds]
        scaff  = [d.scaffold for d in ds]
        frags = lambda idxs: ([smiles[i] for i in idxs], [scaff[i] for i in idxs])
        s_tr, sc_tr = frags(split.train_split)
        s_va, sc_va = frags(split.val_split)
        s_te, sc_te = frags(split.test_split)
        df = pd.DataFrame({
            "split": (["train"]*len(s_tr) + ["val"]*len(s_va) + ["test"]*len(s_te)),
            "smiles": s_tr + s_va + s_te,
            "scaffold": sc_tr + sc_va + sc_te,
        })
        df.to_csv(f"{prefix}_smiles_split_{config.split_index}.csv", index=False)

    dump = getattr(config, "dump_split_csvs", True)
    if dump:
        dump_smiles_scaffolds(activity_ds, split_act,  "activity")
        dump_smiles_scaffolds(pose_ds,     split_pose, "pose")

    print(f"[activity split] train={split_act.train_size} val={split_act.val_size} test={split_act.test_size}")
    print(f"[pose     split] train={split_pose.train_size} val={split_pose.val_size} test={split_pose.test_size}")


    # build Lightning loaders
  
    num_workers = getattr(config, "num_workers", 1)
    dm_activity = make_data_module(
        split_act,
        config.batch_size,
        num_workers=num_workers,
        dataset_instance=activity_ds,
        train_kwargs={"transform": train_transform},
        val_kwargs={"transform": val_transform},
        test_kwargs={"transform": val_transform},
        one_time_transform=one_time_transform,
        normalization=getattr(config, "normalize_activity", True),  # default True
        normalization_idx=act_norm_idx,
    )

    dm_pose = make_data_module(
        split_pose,
        config.batch_size,
        num_workers=num_workers,
        dataset_instance=pose_ds,
        train_kwargs={"transform": train_transform},
        val_kwargs={"transform": val_transform},
        test_kwargs={"transform": val_transform},
        one_time_transform=one_time_transform,
        normalization=getattr(config, "normalize_pose", False),     # default False
    )

    return CombinedDataModule(dm_activity, dm_pose)
