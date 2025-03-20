from typing import Any, Dict, List
from lightning.pytorch.core import LightningDataModule
from torch.utils.data import DataLoader
import webdataset as wds
from transformers import AutoImageProcessor
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
        video_inputs = vision_utils.process_vision_info(video_data_list)
        processed_videos = self.processor(
            images=None,  # Only processing videos here.
            videos=video_inputs,
            return_tensors="pt",
        )

        texts = [sample[key] for sample in batch for key in sample if key.endswith("txt")]

        return {
            "videos": processed_videos,
            "texts": texts,
        }

class DataModule(LightningDataModule):
    def __init__(self, data_dir: str, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.cfg = cfg
        
        self.processor = AutoImageProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        
        self.collator = DataCollatorWithPadding(
            processor=self.processor,
            data_dir=self.data_dir,
            stage=self.cfg["stage"],
            max_length=self.cfg["max_length"]
        )

    def setup(self, stage: str = None) -> None:
        if stage in ("fit", None):
            train_path = os.path.join(self.data_dir, "train",  "{00000..00089}.tar")
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


if __name__ == "__main__":
    # Example cfg input and data_dir

    # Note data_dir must contain train, test, val, and that contains 0000.tar etc. 
    data_dir = "/OpenASL_web/"
    cfg = {"batch_size": 1, 
           "num_workers": 0, 
           "stage": "train", 
           "max_length": 128}
