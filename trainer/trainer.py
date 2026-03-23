from cProfile import label
import os
import sys
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time


class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.is_main_process = int(os.environ.get("RANK", "0")) == 0

    def train(self, dataloader):
        self.model.train()
        stats = {
            'total_loss' : 0.0,
            'processed_samples' : 0
        }
        
        pbar = tqdm(
            dataloader,
            desc='Train',
            bar_format='{l_bar}{bar:30}{r_bar}',
            colour='blue',
            disable=not self.is_main_process,
        )

        for batch in pbar:
            img, label = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            logits = self.model(img)
            total_loss = self.criterion(logits, label)

            total_loss.backward()
            self.optimizer.step()

            batch_size = img.size(0)
            stats['total_loss'] += total_loss.item()
            stats['processed_samples'] += batch_size

            postfix = {
                'loss': f"{stats['total_loss']/stats['processed_samples']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            }

            pbar.set_postfix(postfix)

        return stats['total_loss'] / stats['processed_samples']
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        processed_sample = 0

        pbar = tqdm(
            dataloader,
            desc='val',
            bar_format='{l_bar}{bar:30}{r_bar}',
            colour='green',
            disable=not self.is_main_process,
        )

        with torch.inference_mode():
            for batch in pbar:
                img, label = [x.to(self.device) for x in batch]

                logits = self.model(img)
                loss = self.criterion(logits, label)

                batch_size = img.size(0)
                total_loss += loss.item() * batch_size
                processed_sample += batch_size

                pbar.set_postfix({
                        'loss':f'{total_loss/processed_sample:.4f}'
                    }
                )
        
        return total_loss / processed_sample
