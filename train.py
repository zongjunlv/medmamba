import argparse
import math
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.medical_dataset import Medical_Dataset
from models.model_factory import MODEL_CHOICES, build_model
from trainer import Trainer
from utils.evaluator import evaluate_model

try:
    import swanlab
except ImportError:
    swanlab = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train MedMamba on cached 3D medical volumes.")
    parser.add_argument("--train-csv", required=True, help="Path to training csv.")
    parser.add_argument("--val-csv", required=True, help="Path to validation csv.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay.")
    parser.add_argument("--num-workers", type=int, default=12, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed.")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes.")
    parser.add_argument(
        "--model",
        default="medmamba3d_tiny",
        choices=MODEL_CHOICES,
        help="Backbone used for training.",
    )
    parser.add_argument("--device", default="cuda", help="Device string.")
    parser.add_argument(
        "--save-path",
        default=None,
        help="Path to save the best checkpoint. Defaults to assets/checkpoints_3d/best_model_<model>.pth.",
    )
    parser.add_argument("--swanlab-project", default="MedMamba", help="SwanLab project name.")
    parser.add_argument("--disable-swanlab", action="store_true", help="Disable SwanLab logging.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def warmup_cosine(warmup_epochs, num_epochs, epoch):
    if epoch < warmup_epochs:
        return float(epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_dataloader(dataset, batch_size, shuffle, num_workers, drop_last):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "pin_memory": True,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 8
    return DataLoader(dataset, **kwargs)


def resolve_save_path(save_path_arg, model_name):
    if save_path_arg:
        return Path(save_path_arg)
    return Path("assets/checkpoints_3d") / f"best_model_{model_name}.pth"


def main():
    args = parse_args()
    if swanlab is None:
        args.disable_swanlab = True

    print("\n" + "=" * 60)
    print(f"{'Model Training Pipeline':^60}")
    print("=" * 60)

    start_time = time.time()
    set_seed(args.seed)

    train_path = Path(args.train_csv)
    val_path = Path(args.val_csv)
    if not train_path.is_file():
        raise FileNotFoundError(f"Training csv not found: {train_path}")
    if not val_path.is_file():
        raise FileNotFoundError(f"Validation csv not found: {val_path}")

    train_dataset = Medical_Dataset(mode="train", csv_path=str(train_path))
    val_dataset = Medical_Dataset(mode="val", csv_path=str(val_path))

    train_dataloader = build_dataloader(
        train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )
    val_dataloader = build_dataloader(
        val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False
    )

    device_name = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    device = torch.device(device_name)
    save_path = resolve_save_path(args.save_path, args.model)

    model = build_model(args.model, args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, device)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: warmup_cosine(args.warmup_epochs, args.epochs, epoch)
    )

    print("\nCONFIGURATION")
    print(f"  {'Train csv':18}: {train_path}")
    print(f"  {'Val csv':18}: {val_path}")
    print(f"  {'Batch size':18}: {args.batch_size}")
    print(f"  {'Learning rate':18}: {args.lr}")
    print(f"  {'Num classes':18}: {args.num_classes}")
    print(f"  {'Model':18}: {args.model}")
    print(f"  {'Device':18}: {device}")
    print(f"  {'Save path':18}: {save_path}")

    if not args.disable_swanlab:
        swanlab.init(
            project=args.swanlab_project,
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_classes": args.num_classes,
                "model": args.model,
                "train_csv": str(train_path),
                "val_csv": str(val_path),
                "save_path": str(save_path),
            },
        )

    best_auc = float("-inf")
    early_stop = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = trainer.train(train_dataloader)
        val_loss = trainer.validate(val_dataloader)
        accuracy, auc, sensitivity, specificity, f1, precision, mcc = evaluate_model(
            model,
            val_dataloader,
            device,
            verbose=(epoch % 5 == 0),
        )
        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(
            f"  Train loss: {train_loss:.4f}   Val loss: {val_loss:.4f}   "
            f"Val AUC: {auc:.4f}   Time: {str(timedelta(seconds=int(epoch_time)))}"
        )

        if auc > best_auc:
            best_auc = auc
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  Weights updated! Best AUC: {best_auc:.4f}")
            early_stop = 0
        else:
            early_stop += 1
            print(f"  Early stopping count: {early_stop}/{args.patience}")
            if early_stop >= args.patience:
                print(f"  Early stopping triggered at epoch {epoch + 1}.")
                break

        if not args.disable_swanlab:
            swanlab.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "acc": accuracy,
                    "auc": auc,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "f1": f1,
                    "precision": precision,
                    "mcc": mcc,
                }
            )

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training completed in {str(timedelta(seconds=int(total_time)))}")
    print(f"Best validation AUC: {best_auc:.4f}")
    print("=" * 60 + "\n")

    if not args.disable_swanlab:
        swanlab.finish()


if __name__ == "__main__":
    main()
