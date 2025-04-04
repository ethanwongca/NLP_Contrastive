from typing import Any, Dict, List
from lightning.pytorch.core import LightningDataModule
from torch.utils.data import DataLoader
import webdataset as wds
import torch
from transformers import AutoProcessor
import os 

# Modified Qwen Processor Class
from utils import vision_utils

class DataCollatorWithPadding:
    def __init__(self, processor: Any, data_dir: str, stage:str, max_length: int = 128) -> None:
        self.processor = processor
        self.data_dir = data_dir
        self.stage = stage
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Processing videos via Qwen's Image Processor
        # Extra the bytes from the webdataset as mp4 -> bytes 
        video_data_list = [sample[key] for sample in batch for key in sample if key.endswith("mp4")]
        # Process the raw video data via modified Qwen-VL Utils for the webdataset 
        video_inputs, video_kwargs = vision_utils.process_vision_info(video_data_list, return_video_kwargs=True)

        # Prompt needs the EOS Tokens Etc. 
        messages = [{"role": "user", "content": [{"type": "video"}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        processed_videos = self.processor(
            text=[prompt] * len(video_data_list),
            images=None,  # Only processing videos here.
            videos=video_inputs,
            padding=True, 
            return_tensors="pt",
            **video_kwargs,
        )
        
        # Left padding so we index end of attention mask and extend to the end 
        # Both input_id: (batch_size, sequence_length) and attention_mask: (batch_size, sequence_length)
        num_examples = processed_videos['attention_mask'].shape[0]

        # Set right padding since array is contiguous 
        for i in range(num_examples):
            input_id = processed_videos['input_ids'][i]
            attention_mask = processed_videos['attention_mask'][i]
        
            padding = input_id[attention_mask == 0]
            non_padding = input_id[attention_mask == 1]
        
            new_input_id = torch.cat([non_padding, padding], dim=0)
            new_attention_mask = torch.cat([torch.ones_like(non_padding), torch.zeros_like(padding)], dim=0)
        
            processed_videos['input_ids'][i] = new_input_id
            processed_videos['attention_mask'][i] = new_attention_mask
            
        texts = [sample[key] for sample in batch for key in sample if key.endswith("txt")]
        
        return {
            "videos": processed_videos,
            "texts": texts
        }

class DataModule(LightningDataModule):
    def __init__(self, data_dir: str, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.cfg = cfg
        
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        
        self.collator = DataCollatorWithPadding(
            processor=self.processor,
            data_dir=self.data_dir,
            stage=self.cfg["stage"],
            max_length=self.cfg["max_length"]
        )

    def setup(self, stage: str = None) -> None:
        if stage in ("fit", None):
            train_path = os.path.join(self.data_dir, "train",  "{00000..00031}.tar")
            self.train_dataset = wds.WebDataset(train_path).shuffle(1000)
        if stage in ("validate", None):
            val_path = os.path.join(self.data_dir, "val",  "{00000..00010}.tar")
            self.val_dataset = wds.WebDataset(val_path).shuffle(1000)
        if stage == "predict":
            test_path = os.path.join(self.data_dir, "test",  "{00000..00021}.tar")
            self.test_dataset = wds.WebDataset(test_path).shuffle(1000)
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg["batch_size"],
            num_workers=self.cfg["num_workers"],
            collate_fn=self.collator
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg["batch_size"],
            num_workers=self.cfg["num_workers"],
            collate_fn=self.collator
        )
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg["batch_size"],
            num_workers=self.cfg["num_workers"],
            collate_fn=self.collator
        )