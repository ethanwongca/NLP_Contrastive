import os
import torch
from omegaconf import OmegaConf
from lightning import seed_everything, Trainer
from contrastive_encoders.data import DataModule
from contrastive_encoders.experiment import VideoTextExp
from lightning.pytorch.loggers import WandbLogger
import wandb
from dotenv import load_dotenv
from torch.profiler import profile, schedule, ProfilerActivity
import socket
from datetime import datetime


load_dotenv()  
wandb.login(key=os.getenv("WANDB_API_KEY"))

# Load config from base.yaml
config_path = "config/base.yaml"
config = OmegaConf.load(config_path)


# Setup data module
dm = DataModule(cfg=config.data_cfg)
dm.setup(stage=config.data_cfg.stage)

print("dm set up successfully!")

# Setup model
model = VideoTextExp(
    video_encoder_cfg=config.model_cfg.init_args.video_encoder_cfg,
    text_encoder_cfg=config.model_cfg.init_args.text_encoder_cfg,
    optimizer_cfg = config.model_cfg.init_args.optimizer_cfg,
    loss_cfg = config.model_cfg.init_args.loss_cfg,
    eval_cfg = config.model_cfg.init_args.eval_cfg
)

print("Model Set up successfully!")

wandb_logger = WandbLogger(project="VideoTextExp", name="test_full_ddp", log_model=False) # Setting log_model = False for testing rounds


trainer = Trainer(
    max_epochs=1,
    # limit_train_batches= 12,
    # limit_val_batches= 6,
    val_check_interval=20,
    accelerator="gpu",  
    precision="bf16-mixed",  
    logger=wandb_logger,
    enable_checkpointing=True,
    log_every_n_steps=1,
    devices=1, 
    #strategy="ddp"
    #CUDA_VISIBLE_DEVICES=1
)

trainer.fit(model, datamodule=dm)
