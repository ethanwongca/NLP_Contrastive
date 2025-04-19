'''
Data Module and Collator for Processing Video Data 

Uses WebDataset pipeline to load, shuffle, and batch data 
'''

from typing import Any, Dict, List
from lightning.pytorch.core import LightningDataModule
from torch.utils.data import DataLoader
import webdataset as wds
import torch
from transformers import AutoProcessor
import os 

# Modified Qwen2.5 utils library
from utils import vision_utils

class DataCollatorWithPadding:
    '''
    Custom Collator for padding and processing videos.

    This collator extract video bytes, processes them via our Modified Qwen-Utils,
    and adjusts token sequences by adding a right padding.
    '''
    def __init__(self, processor: Any, data_dir: str, stage:str, max_length: int = 128) -> None:
        '''
        Inputs:
            processor: The Qwen or any LLM Processor
            data_dir: Directory with all the data 
            stage: Either using the train, test, or val data
            max_length: Max Length for Token Sequences 
        '''
        self.processor = processor
        self.data_dir = data_dir
        self.stage = stage
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        '''
        Custom collator for batching
        Input:
            batch: A list of samples
        
        Return: 
            A dictionary of processed videos and raw texts
        '''

        # Video bytes in Webdataset are stored in the mp4 key
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

        # Creates right padding from Qwen's output and cuts the prompt tokens 
        for i in range(num_examples):
            input_id = processed_videos['input_ids'][i]
            attention_mask = processed_videos['attention_mask'][i]
        
            padding = input_id[attention_mask == 0]
            non_padding = input_id[attention_mask == 1]
        
            new_input_id = torch.cat([non_padding, padding], dim=0)
            new_attention_mask = torch.cat([torch.ones_like(non_padding), torch.zeros_like(padding)], dim=0)
        
            # Expected dimensions of input_id (1, 1172)
            # Removes all prompt tokens (Set first Token to be 151652 Qwen2.5-VL, token before video token)
            new_input_id = torch.cat([new_input_id[14::], torch.full((14,), 151643)], dim=0)
            new_attention_mask = torch.cat([new_attention_mask[14::], torch.zeros(14)], dim=0)

            processed_videos['input_ids'][i] = new_input_id
            processed_videos['attention_mask'][i] = new_attention_mask
            
        texts = [sample[key] for sample in batch for key in sample if key.endswith("txt")]
        
        return {
            "videos": processed_videos,
            "texts": texts
        }

class DataModule(LightningDataModule):
    """
    Lightning DataModule for train, val, and predict 
    """
    def __init__(self, data_dir: str, cfg: Dict[str, Any]) -> None:
        """
        Inputs:
            data_dir: Directory with the tar files 
            cfg: Configuration needed to set up data
        """
        super().__init__()
        self.data_dir = data_dir
        self.cfg = cfg
        
        # Qwen 2.5-VL's Processor
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        
        # Custom Collator
        self.collator = DataCollatorWithPadding(
            processor=self.processor,
            data_dir=self.data_dir,
            stage=self.cfg["stage"],
            max_length=self.cfg["max_length"]
        )

    def setup(self, stage: str = None) -> None:
        """Prepare the dataset pipelines for training, validation, or predict"""
        if stage in ("fit", None):
            train_path = os.path.join(self.data_dir, "train", "{00000..00031}.tar") # Need to be set in config
            self.train_dataset = wds.DataPipeline(
                wds.ResampledShards(train_path),  # Sample shards for training.
                wds.shuffle(1000),  # Shuffle shards; num > number in batch
                wds.split_by_worker,  # Split data among DataLoader workers
                wds.tarfile_to_samples(),  # extract the sambles
                wds.batched(
                    self.cfg["batch_size"],
                    collation_fn=lambda x: x
                ),  # Batch samples into list, note data originally in dict
            )
        if stage in ("validate", None):
            val_path = os.path.join(self.data_dir, "val", "{00000..00010}.tar") # Need to be set in config
            self.val_dataset = wds.DataPipeline(
                wds.ResampledShards(val_path),
                wds.shuffle(1000),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.batched(
                    self.cfg["batch_size"],
                    collation_fn=lambda x: x
                ),
            )
        if stage == "predict":
            test_path = os.path.join(self.data_dir, "test", "{00000..00021}.tar") # Need to be set in config
            self.test_dataset = wds.DataPipeline(
                wds.ResampledShards(test_path),
                wds.shuffle(1000),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.batched(
                    self.cfg["batch_size"],
                    collation_fn=lambda x: x
                ),
            )
        
    def train_dataloader(self) -> DataLoader:
        """Return DataLoader for the training dataset"""
        def collate_fn_wrapper(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            return self.collator(batch)

        return DataLoader(
            self.train_dataset,
            batch_size=None,  # Set to None since WDS handles it 
            num_workers=self.cfg["num_workers"],
            collate_fn=collate_fn_wrapper,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return DataLoader for the validation dataset"""
        def collate_fn_wrapper(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            return self.collator(batch)

        return DataLoader(
            self.val_dataset,
            batch_size=None,
            num_workers=self.cfg["num_workers"],
            collate_fn=collate_fn_wrapper,
            pin_memory=True,
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Return DataLoader for the prediction dataset"""
        def collate_fn_wrapper(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            return self.collator(batch)
        
        return DataLoader(
            self.test_dataset,
            batch_size=None,
            num_workers=self.cfg["num_workers"],
            collate_fn=collate_fn_wrapper,
            pin_memory=True,
        )