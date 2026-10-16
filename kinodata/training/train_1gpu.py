#%%
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers.wandb import WandbLogger

from kinodata.data.data_module import make_kinodata_module

import random
import os
import numpy as np


import json
import uuid
from pathlib import Path
from typing import Any


from functools import partial
import sys

# dirty
sys.path.append(".")
sys.path.append("..")


import torch

import kinodata.configuration as cfg
from kinodata.model import ComplexTransformer, RegressionModel
from kinodata.model.complex_transformer import make_model as make_complex_transformer
from kinodata.transform import TransformToComplexGraph

import kinodata.configuration as configuration
from kinodata.model.complex_transformer import ComplexTransformer, make_model
from kinodata.types import NodeType
from kinodata.data.dataset import apply_transform_instance_permament
from kinodata.transform.to_complex_graph import TransformToComplexGraph
import wandb 
from pytorch_lightning.callbacks import ModelCheckpoint


parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, required=True, help="Fold index for cross-validation")
parser.add_argument("--csv_folder_name", type=str, required=True, help="name of csv folder for output")
parser.add_argument("--checkpoint_name", type=str, required=True, help="name of checkpoint folder")
parser.add_argument("--split_type", type=str, default="random-k-fold", choices=["random-k-fold", "scaffold-k-fold", "pocket-k-fold"])
parser.add_argument("--rmsd_max", type=float, default=10, help="RMSD filter (A) applied to both datasets")
parser.add_argument("--max_epochs", type=int, default=500)
# machine-specific settings
parser.add_argument("--num_workers", type=int, default=len(os.sched_getaffinity(0)), help="dataloader workers (default: available CPUs)")
# per-dataset batch size; keep batch_size * accumulate_grad_batches = 128 to match the paper runs
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--accumulate_grad_batches", type=int, default=4)
parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[2] / "runs"), help="root folder for prediction CSVs and checkpoints")
# for smoke tests / timing: int = number of batches, float = fraction
parser.add_argument("--limit_train_batches", type=float, default=1.0)
parser.add_argument("--limit_val_batches", type=float, default=1.0)
parser.add_argument("--limit_test_batches", type=float, default=1.0)
args = parser.parse_args()

def _limit(x):
    return int(x) if x > 1 else x

project_name=f"{args.csv_folder_name}_fold_{args.fold}"
print(f"the project name is {project_name}")

checkpoint_dir = f"{args.output_dir}/{args.checkpoint_name}_fold_{args.fold}"
os.makedirs(checkpoint_dir, exist_ok=True)

# auto-resume: if this run already has a last.ckpt (e.g. the job hit the time limit),
# continue from it and log to the same wandb run
resume_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
resume_ckpt = resume_ckpt if os.path.exists(resume_ckpt) else None
wandb_id_file = os.path.join(checkpoint_dir, "wandb_run_id.txt")
if resume_ckpt and os.path.exists(wandb_id_file):
    wandb_id = Path(wandb_id_file).read_text().strip()
else:
    wandb_id = uuid.uuid4().hex[:8]
    Path(wandb_id_file).write_text(wandb_id)
print(f"resuming from {resume_ckpt}" if resume_ckpt else "starting a new run")

# wandb entity/project/group/mode are taken from the environment:
# WANDB_ENTITY, WANDB_PROJECT, WANDB_RUN_GROUP, WANDB_MODE (online/offline/disabled)
wandb.init(name=project_name, id=wandb_id, resume="allow")



def set_seed(seed=42):
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    
    # Ensure deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # May slow down training but ensures reproducibility
    
    # Set PyTorch Lightning seed
    pl.seed_everything(seed, workers=True)
    
    # Environment variable for dataloader workers
    os.environ["PYTHONHASHSEED"] = str(seed)




####config --> this is the default for training but can be se

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
config["early_stopping_patience"] = 24
config["seed"] = 42
config["split_type"] = args.split_type
config["filter_rmsd_max_value"] = args.rmsd_max
config["k_fold"] = 5
config["split_index"]=args.fold
config["max_epochs"] = args.max_epochs
config["accelerator"] = "gpu"
config["csv_save_dir"]=f"{args.output_dir}/{project_name}"
config["num_workers"] = args.num_workers
set_seed(config["seed"])

print(f"the configuration is {config}")


torch.cuda.empty_cache()
print(torch.cuda.memory_summary())

checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename="best-epoch{epoch:03d}",
    auto_insert_metric_name=False,
    monitor="val/combined_mae",
    mode="min",
    save_top_k=1,
)
class LastCheckpoint(ModelCheckpoint):
    """Writes last.ckpt every epoch for resuming. Unlike ModelCheckpoint it does not
    notify the loggers, so with log_model="all" it is not uploaded to wandb every epoch."""

    def _save_checkpoint(self, trainer, filepath):
        trainer.save_checkpoint(filepath, self.save_weights_only)
        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath


# separate callback so last.ckpt is a real file holding the latest epoch (used for resuming);
# save_last=True would make it a symlink to the best checkpoint, and resuming would overwrite that
last_checkpoint_callback = LastCheckpoint(
    dirpath=checkpoint_dir,
    filename="last",
    save_top_k=1,
    enable_version_counter=False,
)


def train(config, fn_data, fn_model=None):
    
    logger = WandbLogger(
        log_model="all",
        )
    
    model = fn_model(config)# Instantiate the model

   
   
    data_module = fn_data
    logger.watch(model, log="all", log_freq=10, log_graph=True)
    print(data_module)

    # Setup the data module to initialize datasets
    data_module.setup(stage='fit')
    
   
    lr_monitor = LearningRateMonitor("epoch")
    early_stopping = EarlyStopping(
        monitor="val/combined_mae", 
        patience=config.early_stopping_patience, 
        mode="min",
    )
    
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, last_checkpoint_callback, lr_monitor, early_stopping],
        logger=logger,
        max_epochs=config.max_epochs,  
        accelerator=config.accelerator,
        devices=1,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.clip_grad_value,
        limit_train_batches=_limit(args.limit_train_batches),
        limit_val_batches=_limit(args.limit_val_batches),
        limit_test_batches=_limit(args.limit_test_batches),
    )
    if config.dry_run:
        print("Exiting: config.dry_run is set.")
        exit()


    print(f"Max epochs: {trainer.max_epochs}")


    trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt)
    trainer.test(ckpt_path="best", datamodule=data_module)


data_module = make_kinodata_module(
    config,
    transforms=[TransformToComplexGraph(remove_heterogeneous_representation=False)],
)


train(
        config=config,
        fn_model=make_model,
        fn_data=data_module
    )

wandb.finish()
