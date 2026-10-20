from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatchmethod
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.model_selection import GroupKFold, KFold
from collections import Counter
from itertools import islice

from .data_split import Split
from .utils.cluster import AffinityPropagation
from .utils.similarity import BLOSUMSubstitutionSimilarity
from .dataset import KinodataDocked
from .dataset_davids_data import DavidsdataDocked


# ==========
# Small utils
# ==========

def _first_n(it, n=5):
    return list(islice(it, n))

def _split_random(a: np.ndarray, percentile: float, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = default_rng(seed)
    pivot_ = int(a.shape[0] * percentile)
    permuted = rng.permutation(a)
    return permuted[:pivot_], permuted[pivot_:]

def _as_indices(mask_or_idx: np.ndarray, n: int) -> np.ndarray:
    """Normalize boolean mask or integer indices to integer indices."""
    mask_or_idx = np.asarray(mask_or_idx)
    if mask_or_idx.dtype == bool:
        if mask_or_idx.shape[0] != n:
            raise ValueError("Boolean mask length does not match array length.")
        return np.nonzero(mask_or_idx)[0]
    return mask_or_idx.astype(np.int64, copy=False)

def _counts(series_like: np.ndarray) -> pd.Series:
    return pd.Series(series_like, name="g").value_counts()

def limit_group_representation(group_index: np.ndarray,
                               max_samples_per_group: int,
                               seed: int = 0) -> np.ndarray:
    """
    Generic cap per group with seeded sampling (no positional bias).
    Returns indices INTO the original array.
    """
    rng = default_rng(seed)
    group_index = np.asarray(group_index)
    keep: List[int] = []
    for gid, _cnt in _counts(group_index).items():
        idx = np.where(group_index == gid)[0]
        if idx.size <= max_samples_per_group:
            keep.extend(idx.tolist())
        else:
            keep.extend(rng.choice(idx, size=max_samples_per_group, replace=False).tolist())
    return np.array(keep, dtype=np.int64)


# ======================
# Diagnostics & safeguards
# ======================

def _check_group_disjoint(groups: np.ndarray, sp: Split) -> Dict[str, Tuple[int, List]]:
    """Return sizes of group overlaps between splits (train∩val, train∩test)."""
    def as_set(idx):
        return set(groups[idx]) if idx is not None and len(idx) else set()
    T = as_set(sp.train_split)
    V = as_set(sp.val_split)
    S = as_set(sp.test_split)
    leak_tv = T & V
    leak_tt = T & S
    return {
        "train∩val": (len(leak_tv), _first_n(leak_tv)),
        "train∩test": (len(leak_tt), _first_n(leak_tt)),
    }

def assert_no_group_leakage(groups: np.ndarray, splits: List[Split], tag: str) -> None:
    for i, sp in enumerate(splits):
        res = _check_group_disjoint(groups, sp)
        msg_lines = [f"[{tag}] fold {i} group overlaps:"]
        ok = True
        for k, (n, examples) in res.items():
            msg_lines.append(f"  {k}: {n}" + (f" (e.g. {examples})" if n else ""))
            if n > 0:
                ok = False
        if not ok:
            raise RuntimeError("\n".join(msg_lines))
        else:
            print("\n".join(msg_lines))

def print_group_stats_per_fold(groups: np.ndarray, splits: List[Split], tag: str = "") -> None:
    for fold_id, sp in enumerate(splits):
        print(f"\n[{tag}] fold {fold_id}")
        for name, idx in zip(["train", "val", "test"], [sp.train_split, sp.val_split, sp.test_split]):
            part_groups = groups[idx] if idx is not None else np.array([], dtype=groups.dtype)
            cnt = Counter(part_groups)
            largest = cnt.most_common(1)[0][1] if cnt else 0
            print(f"  {name:5s}: {len(cnt):4d} groups, {len(part_groups):5d} samples (largest={largest})")


# =========================
# Core splitting primitives
# =========================

def _generator_to_splits(generator: Iterable[Tuple[np.ndarray, np.ndarray]],
                         val_frac: float = 0.5,
                         seed: int = 0) -> List[Split]:
    """Convert (train_idx, test_idx) folds to Split with inner val/test split."""
    splits: List[Split] = []
    for train_index, test_index in generator:
        val_index, test_index = _split_random(np.asarray(test_index), val_frac, seed)
        splits.append(Split(train_index, val_index, test_index))
    return splits

def group_k_fold_split(group_index: np.ndarray,
                       k: int,
                       max_samples_per_group: Optional[int] = None,
                       seed: int = 0,
                       val_frac: float = 0.5) -> List[Split]:
    """
    GroupKFold with optional per-group downsampling.
    Returns global indices (mapped back after filtering).
    """
    group_index = np.asarray(group_index).ravel()
    n = group_index.shape[0]
    global_idx = np.arange(n)

    if max_samples_per_group is not None:
        keep = limit_group_representation(group_index, max_samples_per_group, seed)
        keep = _as_indices(np.asarray(keep), n)
        group_index_f = group_index[keep]
        global_idx = global_idx[keep]
    else:
        group_index_f = group_index

    n_unique = np.unique(group_index_f).size
    if k > n_unique:
        raise ValueError(f"k={k} is larger than number of unique groups ({n_unique}).")

    gkf = GroupKFold(n_splits=k)
    _X = np.zeros((group_index_f.shape[0], 1))
    gen_local = gkf.split(_X, groups=group_index_f)
    gen_global = ((global_idx[tr], global_idx[te]) for tr, te in gen_local)
    return _generator_to_splits(gen_global, val_frac=val_frac, seed=seed)

def random_k_fold_split(n_samples: int,
                        k: int,
                        seed: int = 0,
                        val_frac: float = 0.5) -> List[Split]:
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    idx = np.arange(n_samples)
    return _generator_to_splits(kf.split(idx), val_frac=val_frac, seed=seed)


# ======================
# Pocket clustering utils
# ======================

def _cap_indices_by_pocket(idx: np.ndarray,
                           groups: np.ndarray,
                           per_pocket_cap: int,
                           seed: int = 0) -> np.ndarray:
    """
    Cap number of samples per pocket (group) *within this index set*.
    idx: integer indices belonging to one split (train/val/test).
    groups: pocket id per sample for the full dataset.
    Returns a NEW array of capped indices (shuffled).
    """
    rng = default_rng(seed)
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return idx

    # bucket indices by pocket (within this split only)
    pockets = {}
    for i in idx:
        pid = groups[i]
        pockets.setdefault(pid, []).append(i)

    kept = []
    for pid, bucket in pockets.items():
        b = np.asarray(bucket, dtype=np.int64)
        rng.shuffle(b)
        kept.extend(b[:per_pocket_cap])

    kept = np.asarray(kept, dtype=np.int64)
    rng.shuffle(kept)  # optional global shuffle
    return kept


def cap_split_by_max_share(split: "Split",
                           groups: np.ndarray,
                           max_share: float,
                           min_cap: int = 1,
                           max_cap: Optional[int] = None,
                           seed: int = 0) -> "Split":
    """
    For each of train/val/test in `split`, compute a per-pocket cap = max_share * |split|.
    Apply that cap so no pocket exceeds this count *within that split*.
    Returns a NEW Split (does not modify input).
    """
    def cap_for(idx):
        n = len(idx) if idx is not None else 0
        base = max(int(np.floor(max_share * n)), min_cap)
        if max_cap is not None:
            base = min(base, max_cap)
        return base

    tr_cap = cap_for(split.train_split)
    va_cap = cap_for(split.val_split)
    te_cap = cap_for(split.test_split)

    tr_new = _cap_indices_by_pocket(split.train_split, groups, tr_cap, seed) if split.train_split is not None else None
    va_new = _cap_indices_by_pocket(split.val_split,   groups, va_cap, seed) if split.val_split   is not None else None
    te_new = _cap_indices_by_pocket(split.test_split,  groups, te_cap, seed) if split.test_split  is not None else None

    return Split(tr_new, va_new, te_new)



def cluster_pockets_indices(n_items: int,
                            pocket_sequences: Iterable[str],
                            clustering: AffinityPropagation,
                            sim_measure: BLOSUMSubstitutionSimilarity) -> np.ndarray:
    """Return group ids (cluster labels) for each item by clustering pocket sequences."""
    df = pd.DataFrame({
        "index": np.arange(n_items),
        "pocket_sequence": np.asarray(list(pocket_sequences)),
    })
    df_labels = clustering(
        df, "pocket_sequence", fn_similarity=sim_measure()
    ).sort_values(by="index", ascending=True)
    assert df_labels.shape[0] == n_items
    return np.asarray(df_labels[clustering.cluster_key].values)


# ===========================
# Public API: single-dataset
# ===========================

class KinodataKFoldSplit:
    """
    Unified splitter for:
      - 'scaffold-k-fold': group by data.scaffold
      - 'pocket-k-fold'  : group by clustered pocket sequences
      - 'random-k-fold'  : no grouping (plain KFold)
    Splits are cached to dataset.processed_dir / split_type / '{i}:{k}.csv'
    """

    pocket_clustering = AffinityPropagation()
    pocket_similarity_measure = BLOSUMSubstitutionSimilarity

    def __init__(self,
                 split_type: str,
                 k: int,
                 *,
                 val_frac: float = 0.5,
                 seed: int = 0,
                 max_samples_per_group: Optional[int] = None,
                 enforce_no_leak: bool = True) -> None:
        assert split_type in ("scaffold-k-fold", "pocket-k-fold", "random-k-fold"), f"Unknown split type {split_type}"
        self.split_type = split_type
        self.k = k
        self.val_frac = val_frac
        self.seed = seed
        self.max_samples_per_group = max_samples_per_group
        self.enforce_no_leak = enforce_no_leak

    @singledispatchmethod
    def cache_dir(self, dataset) -> Path:
        raise NotImplementedError(f"cache_dir({type(dataset)})")

    @cache_dir.register
    def _(self, dataset: KinodataDocked) -> Path:
        return Path(dataset.processed_dir) / self.split_type

    @cache_dir.register
    def _(self, dataset: DavidsdataDocked) -> Path:
        return Path(dataset.processed_dir) / self.split_type

    @cache_dir.register
    def _(self, dataset_dir: Path) -> Path:
        return dataset_dir / self.split_type

    def split_files(self,
                    dataset: Optional[KinodataDocked] = None,
                    dataset_dir: Optional[Path] = None) -> List[Path]:
        cache_dir = self.cache_dir(dataset if dataset is not None else dataset_dir)  # type: ignore[arg-type]
        return [cache_dir / f"{i}:{self.k}.csv" for i in range(1, self.k + 1)]

    def _split(self, dataset) -> List[Split]:
        if self.split_type == "scaffold-k-fold":
            scaffolds = np.array([data.scaffold for data in dataset])
            splits = group_k_fold_split(
                group_index=scaffolds,
                k=self.k,
                #max_samples_per_group=self.max_samples_per_group,
                max_samples_per_group=3000,
                seed=self.seed,
                val_frac=self.val_frac,
            )
            if self.enforce_no_leak:
                print_group_stats_per_fold(scaffolds, splits, tag="scaffold-k-fold")
                assert_no_group_leakage(scaffolds, splits, tag="scaffold-k-fold")
            return splits

        if self.split_type == "pocket-k-fold":
            n_items = len(dataset)
            pockets = cluster_pockets_indices(
                n_items=n_items,
                pocket_sequences=dataset.data.pocket_sequence,
                clustering=self.pocket_clustering,
                sim_measure=self.pocket_similarity_measure,
            )
            splits = group_k_fold_split(
                group_index=pockets,
                k=self.k,
                max_samples_per_group=self.max_samples_per_group,
                seed=self.seed,
                val_frac=self.val_frac,
            )
            if self.enforce_no_leak:
                print_group_stats_per_fold(pockets, splits, tag="pocket-k-fold")
                assert_no_group_leakage(pockets, splits, tag="pocket-k-fold")
            return splits

        # random-k-fold
        return random_k_fold_split(
            n_samples=len(dataset),
            k=self.k,
            seed=self.seed,
            val_frac=self.val_frac,
        )

    def split(self, dataset) -> List[Split]:
        split_files = self.split_files(dataset)
        if all(f.exists() for f in split_files):
            return [Split.from_csv(f) for f in split_files]

        splits: List[Split] = self._split(dataset)

        for split, f in zip(splits, split_files):
            if not f.parents[0].exists():
                f.parents[0].mkdir(parents=True, exist_ok=True)
            split.to_data_frame().to_csv(f, index=False)
            split.source_file = str(f)

        return splits


# =========================================
# Optional: synchronized pocket K-fold dual
# =========================================

@dataclass
class DualSplit:
    activity: Split
    pose: Split

def _cluster_pockets_union(activity_ds,
                           pose_ds,
                           clustering: AffinityPropagation,
                           sim_measure: BLOSUMSubstitutionSimilarity) -> Tuple[np.ndarray, np.ndarray]:
    act_seq = np.asarray(activity_ds.data.pocket_sequence)
    pose_seq = np.asarray(pose_ds.data.pocket_sequence)
    union_df = pd.DataFrame({
        "index": np.arange(len(act_seq) + len(pose_seq)),
        "pocket_sequence": np.concatenate([act_seq, pose_seq], axis=0)
    })
    df_labels = clustering(
        union_df, "pocket_sequence", fn_similarity=sim_measure()
    ).sort_values(by="index", ascending=True)
    labels = np.asarray(df_labels[clustering.cluster_key].values)
    act_labels = labels[:len(act_seq)]
    pose_labels = labels[len(act_seq):]
    return act_labels, pose_labels

def joint_pocket_kfold_split(activity_ds,
                             pose_ds,
                             k: int,
                             *,
                             val_frac: float = 0.5,
                             seed: int = 0,
                             enforce_no_leak: bool = True) -> Tuple[List[DualSplit], np.ndarray, np.ndarray]:
    """
    Produce synchronized K-fold splits by pocket across BOTH datasets.
    No pocket in test appears in train/val in either task.
    """
    act_groups, pose_groups = _cluster_pockets_union(
        activity_ds, pose_ds, KinodataKFoldSplit.pocket_clustering,
        KinodataKFoldSplit.pocket_similarity_measure
    )

    # Concatenate views to drive GroupKFold on shared pocket IDs
    nA, nP = len(act_groups), len(pose_groups)
    groups_all = np.concatenate([act_groups, pose_groups], axis=0)
    origin = np.concatenate([np.zeros(nA, dtype=np.int8), np.ones(nP, dtype=np.int8)], axis=0)
    local_idx = np.concatenate([np.arange(nA), np.arange(nP)], axis=0)

    uniq = np.unique(groups_all).size
    if k > uniq:
        raise ValueError(f"k={k} > unique pockets={uniq}")

    gkf = GroupKFold(n_splits=k)
    X = np.zeros((groups_all.shape[0], 1))
    rng = default_rng(seed)

    folds: List[DualSplit] = []
    for tr_all, te_all in gkf.split(X, groups=groups_all):
        te_perm = rng.permutation(te_all)
        pivot = int(len(te_perm) * val_frac)
        va_all, ts_all = te_perm[:pivot], te_perm[pivot:]

        def sel(orig_flag: int, arr: np.ndarray) -> np.ndarray:
            m = origin[arr] == orig_flag
            return local_idx[arr][m]

        act_tr, act_va, act_ts = sel(0, tr_all), sel(0, va_all), sel(0, ts_all)
        pose_tr, pose_va, pose_ts = sel(1, tr_all), sel(1, va_all), sel(1, ts_all)

        folds.append(DualSplit(
            activity=Split(act_tr, act_va, act_ts),
            pose=Split(pose_tr, pose_va, pose_ts),
        ))

    if enforce_no_leak:
        def _assert_no_leak(groups: np.ndarray, sp: Split, tag: str):
            T = set(groups[sp.train_split]) if sp.train_split is not None else set()
            V = set(groups[sp.val_split])   if sp.val_split is not None   else set()
            S = set(groups[sp.test_split])  if sp.test_split is not None  else set()
            assert not (T & V), f"{tag}: train∩val"
            assert not (T & S), f"{tag}: train∩test"

        for i, dual in enumerate(folds):
            _assert_no_leak(act_groups, dual.activity, f"activity fold {i}")
            _assert_no_leak(pose_groups, dual.pose,     f"pose fold {i}")

    # the pocket group ids are returned too, for optional per-pocket capping in make_kinodata_module
    return folds, act_groups, pose_groups
