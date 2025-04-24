import os
import torch
from omegaconf import OmegaConf
from lightning import seed_everything, Trainer
from contrastive_encoders.data import DataModule
from contrastive_encoders.experiment import VideoTextExp
from lightning.pytorch.loggers import WandbLogger
import wandb
from dotenv import load_dotenv


load_dotenv()  
wandb.login(key=os.getenv("WANDB_API_KEY"))

# Load config from base.yaml
config_path = "config/base.yaml"
config = OmegaConf.load(config_path)

data_dir = "/scratch/st-jzhu71-1/ewong25/my_jupyter/How2Sign_web"

# Setup data module
dm = DataModule(data_dir=data_dir, cfg=config.data_cfg)
dm.setup(stage=config.data_cfg.stage)


# Setup model
model = VideoTextExp(
    video_encoder_cfg=config.model.init_args.video_encoder_cfg,
    text_encoder_cfg=config.model.init_args.text_encoder_cfg,
    loss_cfg = config.model.init_args.loss_cfg
)

wandb_logger = WandbLogger(project="VideoTextExp", name="debug-run", log_model=False) # Setting log_model = False for testing rounds

trainer = Trainer(
    max_epochs=3,
    limit_train_batches= 12,
    limit_val_batches= 6,
    accelerator="gpu",  
    devices=1,
    precision="bf16-mixed",  
    logger=wandb_logger,
    enable_checkpointing=False
)

trainer.fit(model, datamodule=dm)