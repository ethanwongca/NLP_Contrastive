import math

import numpy as np
import torch
import lightning as pl
import transformers
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer
import contrastive_encoders.encoders as encoders
import contrastive_encoders.losses as losses
import sys

import bitsandbytes as bnb

class VideoTextExp(pl.LightningModule):
    def __init__(
         self, 
         video_encoder_cfg,
         text_encoder_cfg,
         optimizer_cfg,
         loss_cfg,
         eval_cfg,
         tokenizer = None,
         processor = None,
     ):
        super().__init__()
        
        self.save_hyperparameters()

        
        self.text_encoder = AutoModel.from_pretrained(
            self.hparams.text_encoder_cfg.pretrained_model_path,
            trust_remote_code=True,
            local_files_only=True # Avoid connecting to HF 
        )
        
        print("text_encoder initialized")

        self.video_encoder = encoders.initialize_vision_encoder(self.hparams.video_encoder_cfg)
        print("video_encoder initialized")

        print(self.hparams)
        self.loss = losses.SigLipLoss(self.hparams.loss_cfg)
        print("loss function initialized")
        
        self.eval_model = SentenceTransformer(self.hparams.eval_cfg.eval_model_path)
        print("eval model initialized")
        
        self.validation_step_outputs = []
        self.video_embeddings = []
        self.text_embeddings = []
        self.video_ids = []
        self.text_ids = []

        
        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if processor is not None:
            self.processor = AutoProcessor.from_pretrained(processor)
            

    def configure_optimizers(self):
        model_params = [
            {"params": self.video_encoder.parameters()},
        ]
        
        
        optimizer = bnb.optim.Adam8bit(model_params, 
                                        lr = self.hparams.optimizer_cfg.initial_lr, 
                                        weight_decay = self.hparams.optimizer_cfg.weight_decay)
        
        max_steps = self.trainer.max_steps 
        
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps = self.hparams.optimizer_cfg.num_warmup_steps,
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

    def validation_step(self, batch, batch_idx, **kwargs):
        video_input = batch["videos"]
        text_input = batch["texts"]

        video_features, text_features = self.forward(video_input, text_input)
        
        self.video_embeddings.append(video_features.cpu())
        self.text_embeddings.append(text_features.cpu())
    
        loss = self.loss(image_features = video_features, 
                        text_features = text_features,
                        logit_scale = math.log(10),
                        logit_bias = -10)
        
        self.validation_step_outputs.append({"val_loss": loss})
        
        self.video_ids.extend([f"vid_{batch_idx}_{j}" for j in range(len(text_input))])
        self.text_ids.extend(text_input)



    def on_validation_epoch_end(self):
        if self.global_rank == 0:
            avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
            self.log("val_loss", avg_loss, sync_dist=True)
            self.validation_step_outputs.clear()
            
            video_embeds = torch.cat(self.video_embeddings, dim=0).numpy()
            text_embeds = torch.cat(self.text_embeddings, dim=0).numpy()
        
            relevant_docs = {
                self.video_ids[i]: {self.text_ids[i]} for i in range(len(self.text_ids))
            }

            # Converting to dictionary dtype to match InformationRetrievalEvaluator
            query_dict = {vid_id: vid_id for vid_id in self.video_ids}
            corpus_dict = {txt_id: txt_id for txt_id in self.text_ids}

            evaluator = InformationRetrievalEvaluator(
                query_dict,
                corpus_dict,
                relevant_docs
            )
            
            accuracy =  evaluator(self.eval_model)
            
            for metric, score in accuracy.items():
                self.log(f"eval/{metric}", score, prog_bar=True, on_epoch=True)
                       
        