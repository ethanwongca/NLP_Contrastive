from typing import Any, Dict, List
from lightning.pytorch.core import LightningDataModule
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoProcessor

class DataCollatorWithPadding:
    def __init__(self, processor: Any, tokenizer: Any, max_length: int = 128) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract video and text data from every sample 
        videos = [sample["mp4"] for sample in batch]
        texts = [sample["txt"] for sample in batch]
        
        # I'M OVERALL UNSURE ABOUT THIS PROCESSOR
        processed_videos = self.processor(videos, return_tensors="pt")
        
        #process texts 
        tokenized_texts = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "videos": processed_videos,
            "texts": tokenized_texts
        }

class DataModule(LightningDataModule):
    def __init__(self, data_dir: str, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.data_dir: str = data_dir
        self.cfg: Dict[str, Any] = cfg
        
        # Load your processors from Hugging Face.
        self.tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        
        self.collator = DataCollatorWithPadding(
            processor=self.processor,
            tokenizer=self.tokenizer,
            max_length=self.cfg["max_length"]
        )

    def setup(self, stage: str = None) -> None:
        if stage in ("fit", None):
            self.train_dataset = load_dataset("webdataset", data_dir=self.data_dir, split="train", streaming=True)
        if stage in ("validate", None):
            self.val_dataset = load_dataset("webdataset", data_dir=self.data_dir, split="val", streaming=True)
        if stage == "predict":
            self.test_dataset = load_dataset("webdataset", data_dir=self.data_dir, split="test", streaming=True)
        
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
    # Example setup for the DataModule + Collator
    cfg: Dict[str, Any] = {
        "batch_size": 4,
        "num_workers": 4,
        "max_length": 128
    }

    data_dir: str = "/path/to/your/tar_folder"
    data_module = DataModule(data_dir, cfg)
    data_module.setup("fit")
