# training/dataset.py
import torch
from torch.utils.data import Dataset
import cv2
import json
import numpy as np

class GTRSDataset(Dataset):
    def __init__(self, root, split="train"):
        self.root = root
        with open(f"{root}/gtrs_filtered.json") as f:
            self.samples = json.load(f)

        self.keys = list(self.samples.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        meta = self.samples[k]

        img = cv2.imread(meta["image"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2,0,1).float()/255.

        depth = np.load(meta["depth"])
        depth = torch.from_numpy(depth).float()

        gtrs = torch.tensor(meta["score"]).float()

        return {
            "image": img,
            "depth": depth,
            "gtrs": gtrs
            }


        """
        return {
            "image": img,
            "depth": depth,
            "gtrs": gtrs,
            "pose": meta.get("pose", None),
            "K": meta.get("K", None)
        }
    """
