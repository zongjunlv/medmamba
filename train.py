import argparse
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run training sequentially for multiple seeds, e.g. --seeds 3407 42 1234.",
    )
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes.")
    parser.add_argument(
        "--model",
        default="medmamba3d_tiny",
        choices=MODEL_CHOICES,
        help="Backbone used for training.",
    )
    parser.add_argument("--device", default="cuda", help="Device string.")
    parser.add_argument(
        "--dist-backend",
        default="nccl",
        help="Distributed backend used when launched with torchrun.",
    )
    parser.add_argument(
        "--find-unused-parameters",
        action="store_true",
        help="Enable DDP unused parameter detection for models with conditional branches.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Path to save the best checkpoint. Defaults to assets/checkpoints_3d/best_model_<model>.pth.",
    )
    parser.add_argument(
        "--log-dir",
        default="assets/training_logs",
        help="Directory to save local training logs and plots.",
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


def build_dataloader(dataset, batch_size, shuffle, num_workers, drop_last, distributed=True):
    sampler = None
    if distributed and dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)

    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "pin_memory": True,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 8
    return DataLoader(dataset, **kwargs)


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def is_main_process():
    return get_rank() == 0


def setup_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return None

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training with torchrun requires CUDA devices.")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.dist_backend)
    return local_rank


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def reduce_mean(value, device):
    if not is_distributed():
        return value
    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor.item()


def resolve_save_path(save_path_arg, model_name):
    if save_path_arg:
        return Path(save_path_arg)
    return Path("assets/checkpoints_3d") / f"best_model_{model_name}.pth"


def resolve_seed_save_path(base_save_path, seed, multi_seed):
    if not multi_seed:
        return base_save_path
    return base_save_path.with_name(f"{base_save_path.stem}_seed{seed}{base_save_path.suffix}")


def resolve_log_paths(log_dir_arg, model_name, seed, multi_seed):
    log_dir = Path(log_dir_arg)
    suffix = f"{model_name}_seed{seed}" if multi_seed else model_name
    csv_path = log_dir / f"train_log_{suffix}.csv"
    plot_path = log_dir / f"train_curves_{suffix}.png"
    return log_dir, csv_path, plot_path


def compute_bias_gap(sensitivity, specificity):
    return abs(float(sensitivity) - float(specificity))


def save_training_artifacts(history, csv_path, plot_path):
    if not history:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv(csv_path, index=False)

    epochs = history_df["epoch"].to_numpy()
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()
    plot_items = [
        ("Loss", [("train_loss", "Train"), ("val_loss", "Val")]),
        ("AUC", [("auc", "Val")]),
        ("Accuracy", [("acc", "Val")]),
        ("F1", [("f1", "Val")]),
        ("Sensitivity", [("sensitivity", "Val")]),
        ("Specificity", [("specificity", "Val")]),
        ("Precision", [("precision", "Val")]),
        ("MCC", [("mcc", "Val")]),
        ("Learning Rate", [("lr", "LR")]),
    ]

    for ax, (title, series_list) in zip(axes, plot_items):
        for column, label in series_list:
            if column in history_df.columns:
                ax.plot(epochs, history_df[column], marker="o", linewidth=1.8, markersize=3, label=label)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.35)
        if len(series_list) > 1:
            ax.legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def train_one_seed(args, seed, train_path, val_path, multi_seed):
    start_time = time.time()
    set_seed(seed)

    if not train_path.is_file():
        raise FileNotFoundError(f"Training csv not found: {train_path}")
    if not val_path.is_file():
        raise FileNotFoundError(f"Validation csv not found: {val_path}")

    train_dataset = Medical_Dataset(mode="train", csv_path=str(train_path), roi_size=(256, 256, 128), margin=12)
    val_dataset = Medical_Dataset(mode="val", csv_path=str(val_path), roi_size=(256, 256, 128), margin=12)

    train_dataloader = build_dataloader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        distributed=True,
    )
    val_dataloader = build_dataloader(
        val_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        distributed=False,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if is_distributed():
        device_name = f"cuda:{local_rank}"
    else:
        device_name = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    device = torch.device(device_name)
    base_save_path = resolve_save_path(args.save_path, args.model)
    save_path = resolve_seed_save_path(base_save_path, seed, multi_seed)
    _, log_csv_path, log_plot_path = resolve_log_paths(args.log_dir, args.model, seed, multi_seed)

    model = build_model(args.model, args.num_classes).to(device)
    if is_distributed():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.find_unused_parameters,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, device)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: warmup_cosine(args.warmup_epochs, args.epochs, epoch)
    )

    if is_main_process():
        print("\nCONFIGURATION")
        print(f"  {'Train csv':18}: {train_path}")
        print(f"  {'Val csv':18}: {val_path}")
        print(f"  {'Batch size':18}: {args.batch_size}")
        print(f"  {'Learning rate':18}: {args.lr}")
        print(f"  {'Num classes':18}: {args.num_classes}")
        print(f"  {'Model':18}: {args.model}")
        print(f"  {'Seed':18}: {seed}")
        print(f"  {'Device':18}: {device}")
        print(f"  {'World size':18}: {dist.get_world_size() if is_distributed() else 1}")
        print(f"  {'Save path':18}: {save_path}")

    if not args.disable_swanlab and is_main_process():
        swanlab.init(
            project=args.swanlab_project,
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_classes": args.num_classes,
                "model": args.model,
                "seed": seed,
                "train_csv": str(train_path),
                "val_csv": str(val_path),
                "save_path": str(save_path),
            },
        )

    best_mcc = float("-inf")
    best_auc_seen = float("-inf")
    early_stop = 0
    history = []

    for epoch in range(args.epochs):
        epoch_start = time.time()
        if is_main_process():
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

        if hasattr(train_dataloader, "sampler") and isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)

        train_loss = trainer.train(train_dataloader)
        train_loss = reduce_mean(train_loss, device)

        if is_main_process():
            val_loss, accuracy, auc, sensitivity, specificity, f1, precision, mcc, _, _, _ = evaluate_model(
                unwrap_model(model),
                val_dataloader,
                device,
                verbose=(epoch % 5 == 0),
                return_outputs=True,
            )
        else:
            val_loss = 0.0
            accuracy = 0.0
            auc = 0.0
            sensitivity = 0.0
            specificity = 0.0
            f1 = 0.0
            precision = 0.0
            mcc = 0.0
        scheduler.step()

        stop_training = False
        if is_main_process():
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]["lr"]
            bias_gap = compute_bias_gap(sensitivity, specificity)
            best_auc_seen = max(best_auc_seen, auc)
            print(
                f"  Train loss: {train_loss:.4f}   Val loss: {val_loss:.4f}   "
                f"Val AUC: {auc:.4f}   Val MCC: {mcc:.4f}   Time: {str(timedelta(seconds=int(epoch_time)))}"
            )
            print(
                f"  Sensitivity: {sensitivity:.4f}   Specificity: {specificity:.4f}   "
                f"Bias gap: {bias_gap:.4f}"
            )

            is_best = mcc > best_mcc
            if is_best:
                best_mcc = mcc
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(unwrap_model(model).state_dict(), save_path)
                print(f"  Weights updated! Best MCC: {best_mcc:.4f}")
                early_stop = 0
            else:
                early_stop += 1
                print(f"  Early stopping count: {early_stop}/{args.patience}")
                if early_stop >= args.patience:
                    print(f"  Early stopping triggered at epoch {epoch + 1}.")
                    stop_training = True

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "acc": accuracy,
                    "auc": auc,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "f1": f1,
                    "precision": precision,
                    "mcc": mcc,
                    "lr": current_lr,
                    "bias_gap": bias_gap,
                    "best_auc_so_far": best_auc_seen,
                    "is_best": int(is_best),
                    "epoch_time_sec": epoch_time,
                }
            )
            save_training_artifacts(history, log_csv_path, log_plot_path)

        if not args.disable_swanlab and is_main_process():
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

        if is_distributed():
            stop_tensor = torch.tensor(int(stop_training), device=device)
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item():
                break
        elif stop_training:
            break

    total_time = time.time() - start_time
    if is_main_process():
        best_metrics = None
        save_training_artifacts(history, log_csv_path, log_plot_path)
        if save_path.is_file():
            best_state_dict = torch.load(save_path, map_location=device, weights_only=True)
            unwrap_model(model).load_state_dict(best_state_dict)
            print("\nBest Checkpoint Evaluation")
            best_metrics = evaluate_model(
                unwrap_model(model),
                val_dataloader,
                device,
                verbose=True,
                show_details=True,
                desc="Best Val",
                return_outputs=False,
            )

        print("\n" + "=" * 60)
        print(f"Training completed in {str(timedelta(seconds=int(total_time)))}")
        print(f"Best validation MCC: {best_mcc:.4f}")
        print(f"Validation AUC upper bound seen during training: {best_auc_seen:.4f}")
        print(f"Training log saved to: {log_csv_path}")
        print(f"Training curves saved to: {log_plot_path}")
        if best_metrics is not None:
            best_val_loss, best_acc, best_auc_eval, best_sens, best_spec, best_f1, best_prec, best_mcc = best_metrics
            print(
                "Best checkpoint metrics: "
                f"loss={best_val_loss:.4f} acc={best_acc:.4f} auc={best_auc_eval:.4f} "
                f"sens={best_sens:.4f} spec={best_spec:.4f} f1={best_f1:.4f} "
                f"prec={best_prec:.4f} mcc={best_mcc:.4f}"
            )
            print(f"Best checkpoint bias gap: {compute_bias_gap(best_sens, best_spec):.4f}")
        print("=" * 60 + "\n")

    if not args.disable_swanlab and is_main_process():
        swanlab.finish()

    return {
        "seed": seed,
        "best_mcc": float(best_mcc),
        "best_auc_seen": float(best_auc_seen),
        "save_path": str(save_path),
        "log_csv_path": str(log_csv_path),
        "log_plot_path": str(log_plot_path),
        "train_time_sec": total_time,
    }


def main():
    args = parse_args()
    if swanlab is None:
        args.disable_swanlab = True

    setup_distributed(args)

    if is_main_process():
        print("\n" + "=" * 60)
        print(f"{'Model Training Pipeline':^60}")
        print("=" * 60)

    train_path = Path(args.train_csv)
    val_path = Path(args.val_csv)
    seeds = args.seeds if args.seeds else [args.seed]
    multi_seed = len(seeds) > 1
    results = []

    for run_idx, seed in enumerate(seeds, start=1):
        if multi_seed:
            print("\n" + "#" * 60)
            print(f"{'Seed Run':^20}: {run_idx}/{len(seeds)}  seed={seed}")
            print("#" * 60)
        result = train_one_seed(args, seed, train_path, val_path, multi_seed)
        if is_main_process():
            results.append(result)

    if multi_seed and is_main_process():
        mccs = np.array([item["best_mcc"] for item in results], dtype=np.float32)
        aucs = np.array([item["best_auc_seen"] for item in results], dtype=np.float32)
        print("\n" + "=" * 60)
        print(f"{'Multi-Seed Summary':^60}")
        print("=" * 60)
        for item in results:
            print(
                f"seed={item['seed']}  best_mcc={item['best_mcc']:.4f}  "
                f"best_auc_seen={item['best_auc_seen']:.4f}  "
                f"checkpoint={item['save_path']}"
            )
        print(f"mean_mcc={mccs.mean():.4f}  std_mcc={mccs.std():.4f}")
        print(f"mean_best_auc_seen={aucs.mean():.4f}  std_best_auc_seen={aucs.std():.4f}")
        print("=" * 60 + "\n")
        summary_path = resolve_save_path(args.save_path, args.model).with_name(
            f"{resolve_save_path(args.save_path, args.model).stem}_multiseed_summary.csv"
        )
        pd.DataFrame(results).to_csv(summary_path, index=False)
        print(f"Multi-seed summary saved to: {summary_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
