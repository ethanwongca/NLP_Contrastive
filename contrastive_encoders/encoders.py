from transformers import AutoProcessor
import torch
import torch.nn as nn
from typing import Optional
from contrastive_encoders.Qwen2_5_vision_encoder import Qwen2_5_VisionTransformerPretrainedModel


from transformers import (Qwen2Tokenizer, 
                          Qwen2VLProcessor, 
                          Qwen2_5_VLPreTrainedModel)
import math

def MeanPooler(hidden_states, attention_mask):
    if attention_mask is not None:
        pooled_output = torch.sum(hidden_states*attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=-1).unsqueeze(-1)
    else:
        pooled_output = torch.mean(hidden_states, dim=1)
    return pooled_output


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        self.projector = nn.Linear(config.truncate_dim, config.proj_size)
        
        
    def forward(self, texts):
        embeddings = self.model.encode(sentences = texts, task=self.config.task, truncate_dim=self.config.out_hidden_size)
        
        tensor_embeddings = torch.tensor(results, dtype=torch.float32) # Converting the embeddings from np array to tensor, shape (batch_size, truncate_dim)
        projected_output = self.projector(tensor_embeddings)
        
        return projected_output

def initialize_text_encoder(cfg):
    model = TextEncoder(cfg)
    return model



class VisionEncoder(Qwen2_5_VLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.encoder = Qwen2_5_VisionTransformerPretrainedModel(config)
        self.projector = nn.Linear(config.out_hidden_size, config.proj_size)
        self.pooler = MeanPooler()
        # Add a pooling layer
    


    def forward(self, video_path, grid_thw, attention_mask: Optional[torch.Tensor] = None):
        
        hidden_states = self.encoder(hidden_states=pixel_values, grid_thw=grid_thw) # (num_tokens, out_hidden_size), haven't try feeding in batch of videos since a single video was already crashing the kernel.
        projected_output = self.projector(hidden_states)
        pooled_output = self.pooler(projected_output,attention_mask) # This may be trouble because the projected_output dimension is (out_hidden_size, proj_size) and the pooler expects (batch_size, num_tokens, truncate_dim) with mean(dim=1).
        
        return hidden_states

def initialize_vision_encoder(cfg):
    model = VisionEncoder(cfg)
    return model

if __name__ == "__main__":
    pass
