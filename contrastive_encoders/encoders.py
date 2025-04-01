from transformers import AutoProcessor, AutoModel, AutoConfig
import torch
import torch.nn as nn
from typing import Optional
from contrastive_encoders.Qwen2_5_vision_encoder import Qwen2_5_VisionTransformerPretrainedModel


from transformers import (Qwen2Tokenizer, 
                          Qwen2VLProcessor, 
                          Qwen2_5_VLPreTrainedModel)
from transformers.models.qwen2_5_vl.modular_qwen2_5_vl import Qwen2_5_VLVisionConfig

import math

def MeanPooler(hidden_states, attention_mask = False):
    if attention_mask is not None:
        pooled_output = torch.sum(hidden_states*attention_mask.unsqueeze(-1), dim=0) / torch.sum(attention_mask, dim=-1).unsqueeze(-1)
    else:
        pooled_output = torch.mean(hidden_states, dim=0, keepdim=True)
    return pooled_output




class VisionEncoder(Qwen2_5_VLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        vision_config = config.vision_config
        self.encoder = Qwen2_5_VisionTransformerPretrainedModel(vision_config)
        print(f"V Encoder config: {self.encoder.config}")
        self.pooler = MeanPooler
        # Add a pooling layer
    


    def forward(self, input_id, pixel_values, grid_thw, attention_mask: Optional[torch.Tensor] = None):
        seq_len = self.encoder.config.seq_len
        out_hidden_size = self.encoder.config.seq_len
        
        hidden_states = self.encoder(hidden_states=pixel_values, grid_thw=grid_thw) # (num_tokens, out_hidden_size)

        input_ids = inputs['input_ids']
        n_video_tokens = (input_ids == config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )
            
        psudo_input_embdeds = torch.zeros(1, seq_len, out_hidden_size)
        video_mask = (
            (input_ids == config.video_token_id)
            .unsqueeze(-1)
            .expand_as(psudo_input_embdeds)
            .to(psudo_input_embdeds.device)
        )
        
        video_embeds = video_embeds.to(psudo_input_embdeds.device, psudo_input_embdeds.dtype)
        
        batched_hidden_states = psudo_input_embdeds.masked_scatter(video_mask, video_embeds)

        attention_mask = torch.zeros(batch_size, seq_len)

        if batch_size == 1:
            attention_mask[0, 15:-7] = 1
        elif batch_size == 2:
            attention_mask[0, 15:] = 1
            attention_mask[1, 7:-7] = 1
        else:
            attention_mask[0, 15:] = 1
            attention_mask[1:-1, 7:] = 1
            attention_mask[-1, 7:-7] = 1

        # expanded_mask = attention_mask.unsqueeze(-1).expand_as(batched_hidden_states)

        # masked_hidden_states = batched_hidden_states * expanded_mask
        
        pooled_output = self.pooler(batched_hidden_states,attention_mask) 
        
        return pooled_output

def initialize_vision_encoder(cfg):
    config = AutoConfig.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
    for key, value in cfg.items():
        setattr(config.vision_config, key, value)
    
    
    model = VisionEncoder(config)
    return model

if __name__ == "__main__":
    pass
