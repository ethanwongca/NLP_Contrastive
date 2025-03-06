import torch
from datasets import load_dataset

from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.core import LightningDataModule

class DataModule(LightningDataModule):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir

    # Setup -> Create datasets etc. 
    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = load_dataset("webdataset", data_dir=self.data_dir, split="train")
        if stage == "validate":    
            self.val_dataset = load_dataset("webdataset", data_dir=self.data_dir, split="val")
        if stage == "predict":
            self.test_dataset = load_dataset("webdataset", data_dir=self.data_dir, split="test")
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=4, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=4, num_workers=4)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=4, num_workers=4)
        