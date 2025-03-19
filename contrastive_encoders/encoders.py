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
        self.pooler = MeanPooler
        # Add a pooling layer
    


    def forward(self, pixel_values, grid_thw, attention_mask: Optional[torch.Tensor] = None):
        
        hidden_states = self.encoder(hidden_states=pixel_values, grid_thw=grid_thw) # (num_tokens, out_hidden_size), haven't try feeding in batch of videos since a single video was already crashing the kernel.
        pooled_output = self.pooler(hidden_states,attention_mask) # This may be trouble because the projected_output dimension is (out_hidden_size, proj_size) and the pooler expects (batch_size, num_tokens, truncate_dim) with mean(dim=1).
        
        return pooled_output

def initialize_vision_encoder(cfg):
    config = AutoConfig.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
    for key, value in cfg.items():
        setattr(config.vision_config, key, value)
    
    
    model = VisionEncoder(config)
    return model

if __name__ == "__main__":
    pass
