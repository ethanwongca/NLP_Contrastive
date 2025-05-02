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