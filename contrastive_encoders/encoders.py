from transformers import AutoConfig
import torch
from contrastive_encoders.Qwen2_5_vision_encoder import Qwen2_5_VisionTransformerPretrainedModel
from transformers import (Qwen2_5_VLPreTrainedModel)



def MeanPooler(hidden_states, attention_mask = None):
    if attention_mask is not None:
        pooled_output = torch.sum(hidden_states*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
    else:
        pooled_output = torch.mean(hidden_states, dim=1, keepdim=False)
    return pooled_output


class VisionEncoder(Qwen2_5_VLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        vision_config = config.vision_config
        self.encoder = Qwen2_5_VisionTransformerPretrainedModel(vision_config)
        # Use pretrained weight first

        self.pooler = MeanPooler
    


    def forward(self, input_ids, pixel_values, grid_thw):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        out_hidden_size = self.encoder.config.out_hidden_size
        
        video_embeds = self.encoder(hidden_states=pixel_values, grid_thw=grid_thw) # (num_tokens, out_hidden_size)

        n_video_tokens = (input_ids == self.encoder.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )
            
        pseudo_input_embdeds = torch.zeros(batch_size, seq_len, out_hidden_size)
        video_mask = (
            (input_ids == self.encoder.config.video_token_id)
            .unsqueeze(-1)
            .expand_as(pseudo_input_embdeds)
            .to(pseudo_input_embdeds.device)
        )

        print(video_mask.shape)
        
        video_embeds = video_embeds.to(pseudo_input_embdeds.device, pseudo_input_embdeds.dtype)
        
        batched_hidden_states = pseudo_input_embdeds.masked_scatter(video_mask, video_embeds)

        pooled_output = self.pooler(
            batched_hidden_states,
            attention_mask=video_mask
        ) 
        
        return pooled_output

def initialize_vision_encoder(cfg):
    vision_encoder_config = AutoConfig.from_pretrained(
        cfg.pretrained_model_path,
        local_files_only=True)
    
    for key, value in cfg.items():
        setattr(vision_encoder_config.vision_config, key, value)
    
    
    model = VisionEncoder(vision_encoder_config)
    return model

if __name__ == "__main__":
    pass
