import torch
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from contrastive_encoders.experiment import VideoTextExp
from contrastive_encoders.data import DataModule
from omegaconf import OmegaConf

# Load config from base.yaml
config_path = "config/base.yaml"
config = OmegaConf.load(config_path)

# Load model
model = VideoTextExp(
    video_encoder_cfg=config.model.init_args.video_encoder_cfg,
    text_encoder_cfg=config.model.init_args.text_encoder_cfg,
    loss_cfg=config.model.init_args.loss_cfg,
)
model.eval()

# Load validation data
data_dir = "How2Sign_web"
dm = DataModule(data_dir=data_dir, cfg=config.data_cfg)
dm.setup(stage="validate")
val_loader = dm.val_dataloader()

# Collect embeddings 
video_embeddings = []
video_ids = []
text_embeddings = []
text_ids = []

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        video_input = batch["videos"]
        text_input = batch["texts"]

        video_embeds, text_embeds = model(video_input, text_input)

        video_embeddings.append(video_embeds)
        text_embeddings.append(torch.tensor(text_embeds))
        
        # vid_i_j corresponds to the ith batch and the jth text sentence
        video_ids.extend([f"vid_{i}_{j}" for j in range(len(text_input))])
        
        # text_ids are the ground truth for the videos
        text_ids.extend(text_input)

# Convert to numpy arrays and move to cpu for SentenceBert
video_embeddings = torch.cat(video_embeddings, dim=0).cpu().numpy()
text_embeddings = torch.cat(text_embeddings, dim=0).cpu().numpy()

# Load evaluator
evaluator = InformationRetrievalEvaluator(
    queries=video_ids,   
    docs=text_ids,    
    relevant_docs={video_ids[i]: {text_ids[i]} for i in range(len(text_ids))},
)

# Run evaluation
evaluator(video_embeddings, text_embeddings, output_path="./eval_results")
