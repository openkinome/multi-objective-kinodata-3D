from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch import Tensor
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import numpy as np
import os
import pandas as pd
from kinodata.configuration import Config
from kinodata.model.resolve import resolve_loss
from kinodata.model.resolve import resolve_optim
from pytorch_lightning.utilities import rank_zero_only

def cat_many(
    data: List[Dict[str, Tensor]], subset: Optional[List[str]] = None, dim: int = 0
) -> Dict[str, Tensor]:
    if subset is None:
        subset = list(data[0].keys())
    assert set(subset).issubset(data[0].keys())

    def ensure_tensor(sub_data, key):
        if isinstance(sub_data[key], torch.Tensor):
            return sub_data[key]
        if isinstance(sub_data[key], list):
            # what have i done
            return torch.tensor([int(x) for x in sub_data[key]])
        raise ValueError(sub_data, key, "cannot convert to tensor")

    return {
        key: torch.cat([ensure_tensor(sub_data, key) for sub_data in data], dim=dim)
        for key in subset
    }



class RegressionModel(pl.LightningModule):

    def __init__(self, config: Config, initial_weight_pki=1, initial_weight_pose=1): 
        super().__init__() 

        self.config = config
        self.save_hyperparameters(config)  # triggers wandb hook
        self.define_metrics()

        self.training_step_outputs = {"activity": [], "pose": []}
        self.test_step_outputs = {"activity": [], "pose": []}
        self.validation_step_outputs = {"activity": [], "pose": []}

        #directory for saving csv files
        self.directory_csv_name = getattr(config, 'csv_save_dir', None)
        if self.directory_csv_name is None:
            raise ValueError("Please specify 'csv_save_dir' in your config file before running the model")
        os.makedirs(self.directory_csv_name, exist_ok=True)


        #  weights
        self.current_weight_pose = initial_weight_pose
        self.current_weight_pki = initial_weight_pki

        # ablation settings; defaults reproduce the paper model
        # activity_eps: floor of the pose weighting of the activity loss (1.0 = unweighted)
        # pose_loss_weight: 0.0 skips the pose batches entirely (no-pose baseline)
        self.activity_eps = getattr(config, "activity_eps", 0.05)
        self.pose_loss_weight = getattr(config, "pose_loss_weight", 1.0)
        self.monitor = getattr(config, "monitor", "val/combined_mae")

              

         
        
    def define_metrics(self):
        wandb.define_metric("val/mae_activity", summary="min")
        wandb.define_metric("val/corr_activity", summary="max")


    def configure_optimizers(self):
        Opt = resolve_optim(self.hparams.optim)
        optim = Opt(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optim,
            mode="min",
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=self.hparams.min_lr,
        )

        return [optim], [
            {
                "scheduler": scheduler,
                "monitor": self.monitor,
                "interval": "epoch",
                "frequency": 1,
            }
        ]
    
    def forward(self, batch) -> Tensor:
        
        pred = self.model(batch)

        return pred
    
    
    def rmsd_to_prob_transform(self, pose_rmsd):
    

        prob_pose = 1 / (1 + torch.exp( 0.9 * (pose_rmsd - 4.5))) 


        
        return prob_pose
    

    def compute_loss_activity(self, pred, batch):

        target_activity = batch.y 

        pred_activity = pred[:, 0] 

        log_uncertainty_act = pred[:, 1]

        pose_pred = pred[:, 2]
        
        

        #gaussian torch loss

        epsilon = self.activity_eps

        variance = torch.exp(log_uncertainty_act)
        pose_certainty=torch.sigmoid(pose_pred).detach()


        loss_fn = torch.nn.GaussianNLLLoss(reduction="none")
        loss_activity_nopose=loss_fn(pred_activity, target_activity, variance)
        
        weights = epsilon + (1 - epsilon) * pose_certainty
        weighted = weights * loss_activity_nopose       
        loss_activity = weighted.sum() / weights.sum().clamp_min(1e-8)

        return loss_activity

       
    def compute_loss_pose(self, pred, batch):

         
        target_exp_rmsd=batch.predicted_rmsd
        
        
        pred_pose_logit=pred[:, 2]


        #converting the input into the sigmoid  
        target_pose_certainty = self.rmsd_to_prob_transform(target_exp_rmsd)
        
        w_low = 10
        w_high = 7
        weight = torch.ones_like(target_pose_certainty)
        weight = torch.where(target_pose_certainty <= 0.2, w_low, weight)
        weight = torch.where(target_pose_certainty >= 0.8, w_high, weight)

        #non-focal loss
        loss_pose=torch.nn.functional.binary_cross_entropy_with_logits(pred_pose_logit, target_pose_certainty, weight=weight)

        return loss_pose




    def training_step(self, batch, batch_idx, dataloader_idx=0, *args) -> Tensor:

        """
        Alternating optimization: Updates one dataset per step.

        """

        loss_activity_raw = torch.tensor(0., device=self.device)
        loss_pose_raw = torch.tensor(0., device=self.device)
        

        if batch["activity"]:  # Dataset 1 (Activity)


            
            batch_activity= batch["activity"]
           

            pred_act = self.forward(batch_activity)


            loss_activity_raw = self.compute_loss_activity(pred_act, batch_activity)
            
            
            self.log("batch_act", batch_activity.num_graphs, batch_size=batch_activity.num_graphs, on_epoch=False, on_step=True)

            

            if self.current_epoch % 10 == 0:
                self.training_step_outputs["activity"].append({
                    "pred_activity": pred_act[:,0].detach().cpu(),
                    "target_activity": batch_activity.y.cpu(),
                    "variance": torch.exp(pred_act[:,1]).detach().cpu(),
                    #"pose":self.rmsd_to_prob_transform(pred_act[:,2]).detach(), 
                    "pose":torch.sigmoid(pred_act[:,2]).detach().cpu(),
                    "target_pose":self.rmsd_to_prob_transform(batch_activity.predicted_rmsd).detach().cpu()
                    })

            

        if batch["pose"] and self.pose_loss_weight > 0:




            batch_pose = batch["pose"]
            
            

            # Forward pass for pose batch
            pred_pose = self.forward(batch_pose) 


            

            loss_pose_raw = self.compute_loss_pose(pred_pose, batch_pose)
            
            self.log("batch_pose", batch_pose.num_graphs, batch_size=batch_pose.num_graphs, on_epoch=False, on_step=True)
            
            if self.current_epoch % 10 == 0:
                self.training_step_outputs["pose"].append({
                    "pose_pred":torch.sigmoid(pred_pose[:,2]).detach().cpu(), 
                    "target_pose":self.rmsd_to_prob_transform(batch_pose.predicted_rmsd).detach().cpu()
                    })


        n_pose=batch["pose"].num_graphs if batch["pose"] is not None else 0
        n_act=batch["activity"].num_graphs

        # standard weighting loss
        activity_loss = self.current_weight_pki * loss_activity_raw 
        pose_loss = self.current_weight_pose * self.pose_loss_weight * loss_pose_raw
        total_loss = activity_loss + pose_loss 
        self.log("train/loss_pose", pose_loss, batch_size=n_pose, on_epoch=True, on_step=True)
        self.log("train/loss_activity", activity_loss, batch_size=n_act, on_epoch=True, on_step=True)
        self.log("train/total_loss", total_loss, batch_size= n_pose+n_act, on_epoch=True, on_step=True)

        self.log("batch_total", n_pose+n_act, batch_size=n_pose+n_act, on_epoch=True, on_step=True)
        
        

        return total_loss
    

    def _reset_buffers(self):
        self.training_step_outputs = {"activity": [], "pose": []}

    def _reset_buffers_val(self):
        self.validation_step_outputs = {"activity": [], "pose": []}

    def _reset_buffers_test(self):
        self.test_step_outputs = {"activity": [], "pose": []}
    
    @rank_zero_only
    def on_train_epoch_end(self):

        if self.current_epoch % 10 != 0: #or self.global_rank != 0:
            self._reset_buffers()

        
        else:

            os.makedirs(self.directory_csv_name, exist_ok=True)

            if self.training_step_outputs["activity"]:
                df_act = pd.concat([pd.DataFrame(d) for d in self.training_step_outputs["activity"]])
                df_act.to_csv(
                    os.path.join(self.directory_csv_name,
                         f"training_activity_epoch_{self.current_epoch}.csv"),
                    index=False
                )

            if self.training_step_outputs["pose"]:
                df_pose = pd.concat([pd.DataFrame(d) for d in self.training_step_outputs["pose"]])
                df_pose.to_csv(
                    os.path.join(self.directory_csv_name,
                         f"training_pose_epoch_{self.current_epoch}.csv"),
                    index=False
                )

            self._reset_buffers()



    
    def validation_step(self, batch, batch_idx, dataloader_idx=0, key: str = "val"):

        """Alternate validation based on dataset type (activity vs pose)."""
 

        if dataloader_idx == 0:  # Dataset 1 (Activity)
            
            
            
            # Forward pass for activity batch
            pred_activity = self.forward(batch)
            act_pred = pred_activity[:, 0]
            pred_log_variance = pred_activity[:, 1]
            pred_variance = torch.exp(pred_log_variance)
            
           


            

                

            self.validation_step_outputs["activity"].append({
                    "pred_activity": act_pred.detach().cpu(),
                    "target_activity": batch.y.cpu(),
                    "variance": pred_variance.detach().cpu(),
                    "pose":torch.sigmoid(pred_activity[:,2]).detach().cpu(),
                    "target_pose":self.rmsd_to_prob_transform(batch.predicted_rmsd).detach().cpu()
            })

        elif dataloader_idx == 1:  # Dataset 2 (Pose)




            # Forward pass for pose batch
            pred_pose_raw = self.forward(batch) 
            pred_pose_logit = pred_pose_raw[:, 2]
            #pred_pose_prob = self.rmsd_to_prob_transform(pred_pose_logit)
            pred_pose_prob = torch.sigmoid(pred_pose_logit) 

            
            
            self.validation_step_outputs["pose"].append({
                    "pred_pose": pred_pose_prob.detach().cpu(),
                    "target_pose": self.rmsd_to_prob_transform(batch.predicted_rmsd).detach().cpu(),
                    })
                


    @rank_zero_only
    def on_validation_epoch_end(self):
        """
        Computes validation metrics at epoch end, handling alternating datasets correctly.
        """

        activity_outputs = self.validation_step_outputs.get("activity", [])


        pred_activity = torch.cat([x["pred_activity"] for x in activity_outputs])
        target_activity = torch.cat([x["target_activity"] for x in activity_outputs])


        activity_mae = (pred_activity- target_activity).abs().mean()

        activity_corr = ((pred_activity - pred_activity.mean()) * (target_activity - target_activity.mean())).mean() / (
                pred_activity.std() * target_activity.std())


        self.log("val/mae_activity", activity_mae, on_epoch=True)
        self.log("val/corr_activity", activity_corr, on_epoch=True)


        if self.validation_step_outputs["pose"]:

            pose_outputs= self.validation_step_outputs.get("pose", [])

            pred_pose = torch.cat([x["pred_pose"] for x in pose_outputs])
            target_pose = torch.cat([x["target_pose"] for x in pose_outputs])

            pose_corr = ((pred_pose - pred_pose.mean()) * (target_pose - target_pose.mean())).mean() / (
                pred_pose.std() * target_pose.std())

            pose_mae = (pred_pose - target_pose).abs().mean()

            self.log("val/mae_pose", pose_mae, on_epoch=True)
            self.log("val/corr_pose", pose_corr, on_epoch=True)


            combined_mae = (activity_mae + pose_mae) / 2
            #combined_mae = activity_mae
            self.log("val/combined_mae", combined_mae, on_epoch=True)


        if self.current_epoch % 10 != 0 :#or self.global_rank != 0:
            self._reset_buffers_val()

        else:    

            os.makedirs(self.directory_csv_name, exist_ok=True)

            if self.validation_step_outputs["activity"]:
                df_act = pd.concat([pd.DataFrame(d) for d in self.validation_step_outputs["activity"]])
                df_act.to_csv(
                    os.path.join(self.directory_csv_name,
                         f"validation_activity_epoch_{self.current_epoch}.csv"),
                    index=False
                )


            if self.validation_step_outputs["pose"]:
                df_pose = pd.concat([pd.DataFrame(d) for d in self.validation_step_outputs["pose"]])
                df_pose.to_csv(
                    os.path.join(self.directory_csv_name,
                         f"validation_pose_epoch_{self.current_epoch}.csv"),
                    index=False
                )

            self._reset_buffers_val()



    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Predict step adapted for alternating dataloaders:
        - dataloader_idx=0: activity dataset
        - dataloader_idx=1: pose dataset
        """

        if dataloader_idx == 0:  # Activity dataset

            pred = self.forward(batch)
            pred_activity = pred[:, 0]
            log_uncertainty = pred[:, 1]
            variance = torch.exp(log_uncertainty)
            pred_pose_logit = pred[:, 2]
            pred_pose = torch.sigmoid(pred_pose_logit)

            target_activity = batch.y
            target_pose = self.rmsd_to_prob_transform(batch.predicted_rmsd)

            return {
                "pred_activity": pred_activity.detach().cpu(),
                "target_activity": target_activity.detach().cpu(),
                "pred_unc_activity": variance.detach().cpu(),
                "pred_pose": pred_pose.detach().cpu(),
                "target_pose": target_pose.detach().cpu(),
            }

        elif dataloader_idx == 1:  # Pose dataset

            pred = self.forward(batch)
            pred_pose_logit = pred[:, 2]
            pred_pose = torch.sigmoid(pred_pose_logit)

            target_pose = self.rmsd_to_prob_transform(batch.predicted_rmsd)

            return {
                "pred_pose": pred_pose.detach().cpu(),
                "target_pose": target_pose.detach().cpu(),
            }

        else:
            raise ValueError(f"Unknown dataloader index: {dataloader_idx}")

      
    def test_step(self, batch, batch_idx, dataloader_idx=0, key: str = "val"):

        """Alternat test based on dataset type (activity vs pose)."""
 

        if dataloader_idx == 0:  # Dataset 1 (Activity)
            
            
            
            # Forward pass for activity batch
            pred_activity = self.forward(batch)
            act_pred = pred_activity[:, 0]
            pred_log_variance = pred_activity[:, 1]
            pred_variance = torch.exp(pred_log_variance)
            
           


            

                

            self.test_step_outputs["activity"].append({
                    "pred_activity": act_pred.detach().cpu(),
                    "target_activity": batch.y.cpu(),
                    "variance": pred_variance.detach().cpu(),
                    "pose":torch.sigmoid(pred_activity[:,2]).detach().cpu(),
                    "target_pose":self.rmsd_to_prob_transform(batch.predicted_rmsd).detach().cpu()
            })

        elif dataloader_idx == 1:  # Dataset 2 (Pose)




            # Forward pass for pose batch
            pred_pose_raw = self.forward(batch) 
            pred_pose_logit = pred_pose_raw[:, 2]
            #pred_pose_prob = self.rmsd_to_prob_transform(pred_pose_logit)
            pred_pose_prob = torch.sigmoid(pred_pose_logit) 
            
            
            self.test_step_outputs["pose"].append({
                    "pred_pose": pred_pose_prob.detach().cpu(),
                    "target_pose": self.rmsd_to_prob_transform(batch.predicted_rmsd).detach().cpu(),
                    })
            

                



    @rank_zero_only
    def on_test_epoch_end(self):
        """
        Computes test metrics at epoch end, handling alternating datasets correctly.
        """

        activity_outputs = self.test_step_outputs.get("activity", [])


        pred_activity = torch.cat([x["pred_activity"] for x in activity_outputs])
        target_activity = torch.cat([x["target_activity"] for x in activity_outputs])


        activity_mae = (pred_activity- target_activity).abs().mean()

        activity_corr = ((pred_activity - pred_activity.mean()) * (target_activity - target_activity.mean())).mean() / (
                pred_activity.std() * target_activity.std())


        self.log("test/mae_activity", activity_mae, on_epoch=True)
        self.log("test/corr_activity", activity_corr, on_epoch=True)


        if self.test_step_outputs["pose"]:

            pose_outputs= self.test_step_outputs.get("pose", [])

            pred_pose = torch.cat([x["pred_pose"] for x in pose_outputs])
            target_pose = torch.cat([x["target_pose"] for x in pose_outputs])

            pose_corr = ((pred_pose - pred_pose.mean()) * (target_pose - target_pose.mean())).mean() / (
                pred_pose.std() * target_pose.std())

            pose_mae = (pred_pose - target_pose).abs().mean()

            self.log("test/mae_pose", pose_mae, on_epoch=True)
            self.log("test/corr_pose", pose_corr, on_epoch=True)


            combined_mae = (activity_mae + pose_mae) / 2
            self.log("test/combined_mae", combined_mae, on_epoch=True)

        os.makedirs(self.directory_csv_name, exist_ok=True)

        if self.test_step_outputs["activity"]:
                df_act = pd.concat([pd.DataFrame(d) for d in self.test_step_outputs["activity"]])
                df_act.to_csv(
                    os.path.join(self.directory_csv_name,
                         f"test_activity_epoch_{self.current_epoch}.csv"),
                    index=False
                )


        if self.test_step_outputs["pose"]:
                df_pose = pd.concat([pd.DataFrame(d) for d in self.test_step_outputs["pose"]])
                df_pose.to_csv(
                    os.path.join(self.directory_csv_name,
                         f"test_pose_epoch_{self.current_epoch}.csv"),
                    index=False
                )



        self._reset_buffers_test()
