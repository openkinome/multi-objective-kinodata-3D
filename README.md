# Multi-objective Kinodata3D

Code and data-processing utilities for **No Pose Left Behind: Integrating Activity and Structural Data with Uncertainty-Aware Multiobjective Learning for Kinase Inhibitor Prediction**.

This repository contains the implementation of `mOKDDD`, a multi-objective E(3)-invariant graph neural network for kinase–ligand binding affinity prediction. The model is designed to learn from experimentally measured activity data paired with *in silico*-generated kinase–ligand complex structures of varying pose quality.

Unlike structure-based models that rely only on poses below a fixed RMSD cutoff, `mOKDDD` explicitly models pose reliability. For each kinase–ligand complex, the model jointly predicts:

- binding affinity, expressed as pIC50;
- activity uncertainty;
- pose quality, expressed as a pose-reliability score q between 0 and 1: a bounded, monotonic transformation of the RMSD to the crystal pose.

The predicted pose quality is used to modulate the contribution of each structure to the activity loss, allowing the model to learn from heterogeneous structural data while giving greater importance to reliable complexes.

## Overview

Structure-based machine learning for kinase inhibitor prediction is limited by the scarcity of experimentally resolved protein–ligand complexes. Computationally generated structures, such as docked poses, can reduce this limitation, but their usefulness depends strongly on pose quality.

`mOKDDD` addresses this by combining activity prediction with pose-quality estimation in a shared E(3)-invariant graph neural network.

![Schematic overview of the multi-objective model workflow](methods_detailed_fig.png)

The workflow consists of four main steps:

**(a) Dataset construction.**  
Two complementary datasets are used during training: [Kinodata](//github.com/openkinome/kinodata), an activity dataset containing kinase–ligand complexes with experimental pIC50 labels and a pose-quality dataset containing generated cross-docked kinase–ligand poses with RMSD-derived pose-quality labels.

**(b) Graph construction and featurization.**  
Each kinase–ligand complex is converted into a molecular graph. Atoms are represented as nodes, while covalent bonds and spatial contacts are represented as edges. The graph is featurized using atom-level descriptors, bond-order information, and interatomic distances.

**(c) E(3)-invariant message passing.**  
Both activity and pose-quality mini-batches are processed by the same E(3)-invariant message-passing GNN.

**(d) Multi-output readout and joint training.**  
A multi-output readout predicts binding affinity, activity uncertainty, and pose quality. These outputs are optimized jointly using a multi-objective loss that combines the activity and pose-quality objectives.

The activity objective learns binding affinity and uncertainty from kinase–ligand complexes with experimental activity labels, while the pose-quality objective learns to predict the pose-reliability score of generated ligand poses. The predicted pose quality is then used to modulate the activity loss, allowing the model to learn from heterogeneous structural data while giving greater importance to reliable complexes.

## Model outputs

For each kinase–ligand complex `x`, the model predicts:

```text
mu(x)         predicted activity
sigma^2(x)   predicted activity variance
q_pose(x)    predicted pose-reliability score
```

## Installation

We currently support installation from source.

### 1. Clone this repository

```
git clone https://github.com/openkinome/multi-objective-kinodata-3D.git
cd multi-objective-kinodata-3D
git checkout v2.0.0   # release that matches the published paper
```

### 2. Set up Python environment
You can use mamba or conda to set up the environment
```
mamba env create -f kinodata_env.yaml
mamba activate kinodata_env
```

Then, install the package with

```
pip install -e .
```

### 3. Obtain raw data
The raw and processed data can be obtained [from Zenodo](https://doi.org/10.5281/zenodo.20433386). After downloading the archives, extract them in the root directory of this repository. See the Zenodo description for more details on the folder structure of the datasets.
The experiments in the paper use the data filtered to an RMSD of at most 10 Å (`data/processed/*/filter_predicted_rmsd_le10.00/`). If only the raw data are present, the processed and filtered datasets are built on first use.

## Reproducing the paper

All experiments use 5-fold cross-validation; each fold is one training run. `kinodata/training/train_1gpu.py` trains one fold, and its defaults are the configuration of the reported models:

| Setting | Value |
|---|---|
| RMSD filter (both datasets) | ≤ 10 Å (`--rmsd_max 10`) |
| Pose-reliability score (target) | q = 1 / (1 + exp(0.9 (RMSD − 4.5))) |
| Pose loss | BCE with class weights 10 (q ≤ 0.2) and 7 (q ≥ 0.8), 1 otherwise |
| Activity loss | Gaussian NLL weighted by ε + (1 − ε) q, ε = 0.10 (`--activity_eps 0.10`) |
| Total loss | L_activity + L_pose |
| Batches | 32 activity + 32 pose complexes per step, gradients accumulated over 4 steps (effective batch 128 per task) |
| Optimizer | Adam, learning rate 3e-4, weight decay 1e-3; learning rate × 0.9 after 8 epochs without improvement (min 1e-6) |
| Model selection | early stopping (patience 24) on the validation mean of activity and pose MAE; at most 500 epochs; the best checkpoint is tested |
| Splits | `random-k-fold`, `scaffold-k-fold` (joint Bemis–Murcko scaffold groups for both datasets, at most 3,000 poses per scaffold), `pocket-k-fold` (cold target: joint KLIFS pocket-sequence clusters); the held-out fold is split in half into validation and test (80/10/10) |
| Seeds | training seed 42 (`--seed`), split seed 0 |

### Training commands

Run from `kinodata/training`; outputs go to `runs/` in the repository root (`--output_dir` to change it).

```bash
cd kinodata/training
export WANDB_MODE=offline        # or online / disabled; WANDB_ENTITY and WANDB_PROJECT select the wandb project

for SPLIT in random-k-fold scaffold-k-fold pocket-k-fold; do
  for FOLD in 0 1 2 3 4; do
    # pose-aware model
    python train_1gpu.py --fold $FOLD --split_type $SPLIT \
        --csv_folder_name pose_aware_$SPLIT --checkpoint_name checkpoints_pose_aware_$SPLIT
    # no-pose baseline: activity Gaussian NLL only (unweighted), no pose objective, model selection on activity MAE
    python train_1gpu.py --fold $FOLD --split_type $SPLIT --no_pose \
        --csv_folder_name no_pose_$SPLIT --checkpoint_name checkpoints_no_pose_$SPLIT
  done
done
```

Each run writes to `runs/<csv_folder_name>_fold_<k>/` the split (`activity_smiles_split_<k>.csv`, `pose_smiles_split_<k>.csv`), per-sample predictions of the training and validation sets every 10 epochs and of the test set with the best checkpoint (`test_activity_epoch_<e>.csv`: `pred_activity`, `target_activity`, `variance`, `pose`, `target_pose`; `test_pose_epoch_<e>.csv`). Activities are z-scored with the training mean and standard deviation, so predictions and targets are in z-units. Checkpoints go to `runs/<checkpoint_name>_fold_<k>/`. A run that is interrupted continues from `last.ckpt` when started again with the same names.

Training one fold takes about 1.5 days on an NVIDIA L40S (~25 min per epoch). The default batch size needs more than 12 GB of GPU memory. With Slurm, `kinodata/training/train_fold.sbatch` runs one fold; its header shows the usage.

### Ablations

| Model | Extra arguments |
|---|---|
| Pose task without pose-dependent weighting of the activity loss | `--activity_eps 1` |
| Pose training data restricted to RMSD ≤ X Å (activity data unchanged) | `--train_rmsd_max_pose X` |
| Both datasets restricted to RMSD ≤ X Å ("good poses only") | `--train_rmsd_max_pose X --train_rmsd_max_activity X` |

The RMSD restrictions apply to the training and validation sets only; the test sets and the activity normalization are those of the unrestricted split, so all models are evaluated on the same test complexes.

### Analysis

`analysis_publication/` contains notebooks for most of the analyses in the paper; not all analyses are included yet. Each notebook reads the small data files it needs from `analysis_publication/data/` (test predictions, per-fold values, RMSD tables), so it runs from this repository without the training data or the trained models. Figures are written to `analysis_publication/output/`.

| Notebook | Content |
|---|---|
| `overall_performance.ipynb` | Test MAE and Pearson correlation per fold of the activity and pose-reliability predictions, random and scaffold splits (Fig. 5) |
| `pose_reliability_validation.ipynb` | Predicted pose-reliability score on cross-docked test poses with a crystal reference: confusion matrices (Fig. 6), predicted uncertainty vs true RMSD |
| `rmsd_distribution.ipynb` | RMSD distributions of both datasets, effect of the RMSD cutoff, pose-reliability score as a function of RMSD |
| `dataset_overlap.ipynb` | Shared ligands, scaffolds, kinases and kinase–ligand pairs between the activity and the pose dataset; train/test assignment across tasks |
| `gradient_balance.ipynb` (+ `gradient_balance.py`) | Balance between the activity and pose-quality objectives: number of updates, gradient norms and alignment on the shared network |
| `ablation_pose_range.ipynb` | Ablations removing one part of the pose-aware training at a time (ε = 1, pose data restricted by RMSD, good poses only) |
| `ablation_good_poses.ipynb` | Training on good poses only vs keeping and down-weighting low-quality poses |
| `similarity_to_training.ipynb` | Accuracy and uncertainty calibration as a function of the Tanimoto similarity to the training set |
| `cold_target.ipynb` | Cold-target split: performance on kinases not seen in training |

`analysis_publication/plot_style.py` holds the plot style of the paper figures.
