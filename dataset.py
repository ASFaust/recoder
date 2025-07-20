import os
import requests
import numpy as np
import torch
from torch.utils.data import Dataset

def download_enwik8():
    url = "http://mattmahoney.net/dc/enwik8.zip"
    if not os.path.exists("enwik8.zip"):
        print("Downloading enwik8...")
        r = requests.get(url)
        with open("enwik8.zip", "wb") as f:
            f.write(r.content)
    if not os.path.exists("enwik8"):
        import zipfile
        with zipfile.ZipFile("enwik8.zip", "r") as zip_ref:
            zip_ref.extractall(".")
    with open("enwik8", "rb") as f:
        data = f.read(10_000_000)  # 10MB
    return np.frombuffer(data, dtype=np.uint8)

class ByteSequenceDataset(Dataset):
    def __init__(self, data, block_size=512):
        self.data = torch.tensor(data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        return self.data[idx:idx + self.block_size]
