import os
from sched import scheduler
import time
import math
import random
from matplotlib.style import available

import numpy as np
import pandas as pd
import swanlab
from datetime import timedelta
import torch
from timm.models.layers import weight_init
from torch.cuda import Device
from torch.optim import optimizer
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn


from models.medmamba3d_videomamba import create_medmamba3d_tiny
from trainer import Trainer
from utils import compute_all_metrics
from data.medical_dataset import Medical_Dataset
from models import model
from utils.evaluator import evaluate_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def warmup_cosine(warmup_epochs, num_epochs, epoch):
    if epoch < warmup_epochs:
        return float(epoch + 1) / warmup_epochs  # 逐步升到基础 lr
    progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine 下降到 0

def main():

    print("\n" + "="*60)
    print(f"{'Model Training Pipeline':^60}")
    print("="*60)

    start_time = time.time()
    seed = 3407
    set_seed(seed)

    # train_path = '/data02/workspace/LZJ_SPACE/MedMamba/dataset/ABUS_Classification/ABUS_NEW/split_train_7_1_2.csv'
    # val_path = '/data02/workspace/LZJ_SPACE/MedMamba/dataset/ABUS_Classification/ABUS_NEW/split_val_7_1_2.csv'
    # test_path = '/data02/workspace/LZJ_SPACE/MedMamba/dataset/ABUS_Classification/ABUS_NEW/split_test_7_1_2.csv'

    train_path = '/data02/workspace/LZJ_SPACE/MedMamba/dataset/ABUS_Classification/TDSC/labels_train_cache.csv'
    val_path = '/data02/workspace/LZJ_SPACE/MedMamba/dataset/ABUS_Classification/TDSC/labels_val_cache.csv'
    
    train_dataset = Medical_Dataset(mode='train', csv_path= train_path)
    val_dataset = Medical_Dataset(mode='val', csv_path= val_path)
    

    batch_size = 4
    lr = 1e-5
    weight_decay = 1e-2

    # 读取训练集标签统计（按文件，标签 0/1/2）
    train_df = pd.read_csv(train_path)
    # counts = train_df['label'].value_counts().reindex([0,1,2], fill_value=1).astype(float)
    num_classes = 2
    counts = train_df['label'].value_counts().reindex([0,1], fill_value=1).astype(float)
    # 采样权重：使用 sqrt(1/freq) 以避免过度倾向少数类
    weight_map = (counts.max() / counts) ** 0.5
    weight_map = weight_map / weight_map.sum()
    train_labels = train_df['label'].tolist()
    sample_weights = [weight_map[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


    train_dataloader = DataLoader(train_dataset, 
                                  batch_size, 
                                  shuffle=True, 
                                #   sampler=sampler,
                                  pin_memory=True, 
                                  drop_last=True,                                        num_workers=12,
                                  persistent_workers=True, 
                                  prefetch_factor=8)
    
    val_dataloader = DataLoader(val_dataset, 
                                batch_size, 
                                shuffle=False, 
                                pin_memory=True,                                        num_workers=12,
                                persistent_workers=True, 
                                prefetch_factor=8)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # model = Model()

    model = create_medmamba3d_tiny(num_classes=num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay )

    
    # 反比权重，防止过大可以再做归一化
    weights_cls = (counts.max() / counts) ** 0.5   # 开平方降温
    weights_cls = weights_cls / weights_cls.mean()
    class_weights = torch.tensor(weights_cls.values, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss()

    print("\nCONFIGURATION")
    print(f"  {'Batch size':18}: {batch_size}")
    print(f"  {'Learning rate':18}: {lr}")
    print(f"  {'Device':18}: {device}")
    print(f"  {'Mixed Precision':18}: {'Unabled' if device.type == 'cuda' else 'Disabled (CPU)'}")

    print("\nPREPARING DATA...")


    trainer = Trainer(model, optimizer, criterion, device)                                      

    warmup_epochs = 5
    num_epochs = 200
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: warmup_cosine(warmup_epochs, num_epochs, epoch)
    )
    best_auc = float('-inf')
    early_stop = 0
    patience = 20

    print("\n" + "-"*60)
    print(f"{'TRAINING STARTED':^60}")
    print("-"*60)
    
    swanlab.init(
    # 设置项目名
        project="my-awesome-project",
        
        # 设置超参数
        config={
            "learning_rate": lr,
            "architecture": "ABUS",
            "epochs": num_epochs,
            "logdir": "/data02/workspace/LZJ_SPACE/MedMamba/assets/logs"
        }
    )
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss = trainer.train(train_dataloader)
        val_loss = trainer.validate(val_dataloader)
        if epoch % 5 == 0:
            accuracy, auc, sensitivity, specificity, f1, precision, mcc = evaluate_model(model, val_dataloader, device, verbose=True)
        else:
            accuracy, auc, sensitivity, specificity, f1, precision, mcc = evaluate_model(model, val_dataloader, device, verbose=False)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f"  Train loss: {train_loss:.4f}   Val loss: {val_loss:.4f}   Val AUC: {auc:.4f}   Time: {str(timedelta(seconds=int(epoch_time)))}")

        if auc > best_auc:
            best_auc = auc
            save_dir = 'assets/checkpoints_3d'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  Weights updated! Best AUC: {best_auc:.4f}")
            early_stop = 0
        else:
            early_stop += 1
            print(f"  Early stopping count: {early_stop}/{patience}")
            if early_stop == patience:
                print(f"  Early stopping triggered at epoch {epoch + 1}.")
                print(f"  Best model saved as: best_model.pth")
                break

        swanlab.log({"train_loss":train_loss,
                    "val_loss": val_loss, 
                    "acc":accuracy,
                    "auc":auc, 
                    "sensitivity":sensitivity, 
                    "specificity":specificity, 
                    "f1":f1,
                    "precision":precision, 
                    "mcc":mcc} 
        )
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"Training completed in {str(timedelta(seconds=int(total_time)))}")
    print(f"Best validation AUC: {best_auc:.4f}")
    print("="*60 + "\n")
    swanlab.finish()
                


if __name__ == '__main__':
    main()
