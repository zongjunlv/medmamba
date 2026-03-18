import os
import numpy as np
import pandas as pd

import torch
from torch.utils import data
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, RandRotate90, RandFlip, RandGaussianNoise, ScaleIntensity)



class Medical_Dataset(Dataset):
    def __init__(self,  mode, csv_path):
        self.mode = mode
        self.data = pd.read_csv(csv_path)
        self.train_aug = Compose([
            ScaleIntensity(minv=0.0, maxv=1.0),
            # RandFlip(prob=0.5, spatial_axis=0),
            # RandFlip(prob=0.5, spatial_axis=1),
            # RandFlip(prob=0.5, spatial_axis=2),
            # RandRotate90(prob=0.5, spatial_axes=(0, 1)),
            # RandRotate90(prob=0.5, spatial_axes=(0, 2)),
        ])
        self.val_aug = Compose([ScaleIntensity(minv=0.0, maxv=1.0)])
        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        row = self.data.iloc[idx]
        img = np.load(row["npy_path"])
        # 确保有通道维度：原始保存为 (D,H,W) 时补成 (1,D,H,W)
        if img.ndim == 3:
            img = img[None, ...]
        if self.mode == 'train':
            img = self.train_aug(img)
        else:
            img = self.val_aug(img)
        img = torch.as_tensor(img).float()
        label = torch.as_tensor(int(row["label"]), dtype=torch.long)

        return img, label
