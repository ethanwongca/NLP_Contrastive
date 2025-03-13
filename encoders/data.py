from typing import Any, Dict, List
from lightning.pytorch.core import LightningDataModule
from torch.utils.data import DataLoader
import webdataset as wds
from transformers import AutoTokenizer, AutoProcessor
import os 

# Modified Qwen Processor Class
from utils import vision_utils


class DataCollatorWithPadding:
    def __init__(self, processor: Any, tokenizer: Any, data_dir: str, max_length: int = 128) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Processing videos according to Qwen's Processor 
        video_paths = [self.data_dir + sample["mp4"] for sample in batch]

        # Vision Inputs 
        video_inputs = vision_utils.process_vision_info(video_paths)

        processed_videos = self.processor(
            text=None,  # Only processing videos here.
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Process texts with Jina Toeknizer 
        texts = [sample["txt"] for sample in batch]
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
        self.data_dir = data_dir
        self.cfg = cfg
        
        # Load processoors seperately 
        self.tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        
        self.collator = DataCollatorWithPadding(
            processor=self.processor,
            tokenizer=self.tokenizer,
            data_dir=self.data_dir,
            max_length=self.cfg["max_length"]
        )

    def setup(self, stage: str = None) -> None:
        if stage in ("fit", None):
            train_path = os.path.join(self.data_dir, "train",  "*.tar")
            self.train_dataset = wds.WebDataset(train_path).shuffle(1000)
        if stage in ("validate", None):
            val_path = os.path.join(self.data_dir, "val",  "*.tar")
            self.val_dataset = wds.WebDataset(val_path).shuffle(1000)
        if stage == "predict":
            test_path = os.path.join(self.data_dir, "test",  "*.tar")
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
    cfg: Dict[str, Any] = {
        "batch_size": 2,  # Start small for testing.
        "num_workers": 0,  # Easier debugging.
        "max_length": 128
    }

    data_dir: str = "/path/to/your/tar_folder"
    
    # Test vision processor.
    print("Testing vision processor...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    test_video = [os.path.join(data_dir, "sample.mp4")]
    vision_output = processor(videos=test_video, return_tensors="pt")
    print("Vision output shapes:")
    print(f"Pixel values: {vision_output.get('pixel_values_videos', vision_output.get('pixel_values')).shape}")
    print(f"Grid specs: {vision_output.get('video_grid_thw')}")

    print("\nTesting tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
    test_text = ["A sample caption"]
    text_output = tokenizer(test_text, padding=True, return_tensors="pt")
    print("Text output keys:", text_output.keys())

    print("\nTesting full collator...")
    data_module = DataModule(data_dir, cfg)
    data_module.setup("fit")
    
    sample_batch = next(iter(data_module.train_dataloader()))
    print("\nFinal batch structure:")
    print("Video keys:", sample_batch["videos"].keys())
    # Adjust key names if necessary based on your processor's output.
    video_tensor = sample_batch["videos"].get("pixel_values", None)
    if video_tensor is not None:
        print(f"Video tensor shape: {video_tensor.shape}")
    print(f"Text input_ids shape: {sample_batch['texts']['input_ids'].shape}")