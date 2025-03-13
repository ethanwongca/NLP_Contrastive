import math

import numpy as np
import torch
import lightning as pl
import transformers
from transformers import AutoTokenizer

import contrastive_encoders.encoders as encoders
import contrastive_encoders.losses as losses

import bitsandbytes as bnb


class VideoTextExp(pl.LightningModule):
    def __init__(
        self, 
        video_encoder_cfg,
        text_encoder_cfg,
        #optimizer,
        sample_rate: int = 16000,
        initial_lr: float = 1e-4,
        weight_decay: float = 1e-4,
        num_warmup_steps: int = 0,
        hard_negatives: bool = False,
        tokenizer = None,
        processor = None,
        text = False
    ):
        super().__init__()

        self.save_hyperparameters()
        
        #print(self.hparams.text_encoder_cfg)
        #print(self.hparams.video_encoder_cfg)
        self.text_encoder = encoders.initialize_text_encoder(self.hparams.text_encoder_cfg)
        print("text_encoder initiated")
        self.video_encoder = encoders.initialize_vision_encoder(self.hparams.video_encoder_cfg)
        print("video_encoder initiated")
        
        self.loss = losses.ContrastiveSigmoid
        
        # t and b for the loss function, should be set in the yaml file?
        self.t_prime = torch.tensor(math.log(10))
        self.b = torch.tensor(-10.0)
        
        self.hard_negatives = hard_negatives
        self.text = text 
        self.validation_step_outputs = []
        
        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if processor is not None:
            self.processor = AutoProcessor.from_pretrained(processor)
            
        #self.zeroshot = zeroshot 

    def configure_optimizers(self):
        model_params = [
            {"params": self.video_encoder.parameters()},
            {"params": self.text_encoder.parameters()}
        ]
        
        # Using 8-bit adam
        optimizer = bnb.optim.Adam8bit(model_params, lr = self.hparams.initial_lr, weight_decay = self.hparams.weight_decay)

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
        video_features = self.encode_video(x) 
        text_features = self.text_encoder(x)
        return video_features, text_features

    def encode_video(self, video_input):
        video_features = self.video_encoder(**video_input).pooler_output
        return video_features

    def encode_text(self, text_input):
        text_features = self.text_encoder(**text_input)
        return text_features

    def training_step(self, batch, batch_idx):
        
        video_input, text_input = batch
        video_features, text_features = self.forward(video_input, text_input)
        loss = self.loss(video_features, 
                         text_features, 
                         self.t_prime,
                         self.b
                        )
    
        self.log("loss", loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        video_input, text_input = batch
        video_features, text_features = self.forward(video_input, text_input)
        loss = self.loss(video_features, 
                         text_features, 
                         self.t_prime,
                         self.b
                        )
        
        self.validation_step_outputs.append(loss)


    def validation_epoch_end(self, outputs):
        
        if self.global_rank == 0:
            avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
            self.log("val_loss", avg_loss, sync_dist = True)

    