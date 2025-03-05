from transformers import AutoProcessor
import torch
import torch.nn as nn

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions
)

from transformers import Qwen2Tokenizer, Qwen2VLProcessor
from PIL import Image
import math

def MeanPoller(hidden_states, attention_mask):
    if attention_mask is not None:
        pooled_output = torch.sum(hidden_states*attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=-1).unsqueeze(-1)
    else:
        pooled_output = torch.mean(hidden_states, dim=1)
    return pooled_output


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        self.projector = nn.Linear(config.truncate_dim, config.proj_size)
        self.pooler = MeanPooler
        
        
    def forward(self, texts):
        embeddings = self.model.encode(texts = texts, task = config.task, truncate_dim=config.truncate_dim)
        
        # Since the projected output is of dimension (batchsize, proj_size), we don't need to use the MeanPooler for this?
        projected_output = self.projector(embeddings)
        
        return projected_output

def initialize_text_encoder(cfg):
    model = TextEncoder(cfg)
    return model



class VideoEncoder(nn.Module):
    