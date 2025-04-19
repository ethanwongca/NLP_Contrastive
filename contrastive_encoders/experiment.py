import math

import numpy as np
import torch
import lightning as pl
import transformers
from transformers import AutoTokenizer, AutoModel, AutoProcessor

import contrastive_encoders.encoders as encoders
import contrastive_encoders.losses as losses

import bitsandbytes as bnb


class VideoTextExp(pl.LightningModule):
    def __init__(
         self, 
         video_encoder_cfg,
         text_encoder_cfg,
         loss_cfg,
         #optimizer,
         sample_rate: int = 16000,
         initial_lr: float = 1e-4,
         weight_decay: float = 1e-4,
         num_warmup_steps: int = 0,
         tokenizer = None,
         processor = None,
         text = False
     ):
        super().__init__()

        self.save_hyperparameters()

        self.text_encoder = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", 
                                                      trust_remote_code=True)
        print("text_encoder initialized")

        self.video_encoder = encoders.initialize_vision_encoder(self.hparams.video_encoder_cfg)
        print("video_encoder initialized")

        print(self.hparams)
        self.loss = losses.SigLipLoss(self.hparams.loss_cfg)
        print("loss function initialized")
        
        self.validation_step_outputs = []
        
        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if processor is not None:
            self.processor = AutoProcessor.from_pretrained(processor)
            

    def configure_optimizers(self):
        model_params = [
            {"params": self.video_encoder.parameters()},
        ]
        
        
        optimizer = bnb.optim.Adam8bit(model_params, 
                                        lr = self.hparams.initial_lr, 
                                        weight_decay = self.hparams.weight_decay)
        
        max_steps = self.trainer.max_steps 
        
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps = self.hparams.num_warmup_steps,
            num_training_steps = max_steps
        )
        
        return(
            [optimizer],
            [{"scheduler": scheduler, "interval": "step"}]
        )
        

    def forward(self, video_input, text_input):
        if isinstance(text_input[0], bytes):
            text_input = [t.decode("utf-8") for t in text_input]
        
        text_features = self.text_encoder.encode(text_input, 
                                                 task=self.hparams.text_encoder_cfg.task, 
                                                 truncate_dim=self.hparams.text_encoder_cfg.out_hidden_size)
        text_features = torch.tensor(text_features)
        
        video_features = self.encode_video(video_input) 
        return video_features, text_features

    def encode_video(self, video_input):
        input_ids = video_input['input_ids']
        pixel_values = video_input['pixel_values_videos']
        grid_thw = video_input['video_grid_thw']
        
        video_features = self.video_encoder(input_ids, pixel_values, grid_thw)

        return video_features

    def encode_text(self, text_input):
        text_features = self.text_encoder(**text_input)
        return text_features

    def training_step(self, batch):
        
        video_input = batch["videos"]
        text_input = batch["texts"]
        video_features, text_features = self.forward(video_input, text_input)
        loss = self.loss(image_features = video_features, 
                        text_features = text_features,
                        logit_scale = math.log(10),
                        logit_bias = -10)
    
        self.log("loss", loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch):
        video_input, text_input = batch
        video_features, text_features = self.forward(video_input, text_input)
        loss = self.loss(video_features, 
                         text_features
                        )
        
        self.validation_step_outputs.append(loss)


    def on_validation_epoch_end(self):
        if self.global_rank == 0:
            avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
            self.log("val_loss", avg_loss, sync_dist=True)
            # Important: Clear the list for the next epoch
            self.validation_step_outputs.clear()