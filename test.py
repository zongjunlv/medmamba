import argparse
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.medical_dataset import Medical_Dataset
from models.model_factory import MODEL_CHOICES, build_model
from utils.evaluator import evaluate_model

try:
    import swanlab
except ImportError:
    swanlab = None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a 3D medical image classifier.")
    parser.add_argument("--test-csv", required=True, help="Path to test csv.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint. Defaults to assets/checkpoints_3d/best_model_<model>.pth.",
    )
    parser.add_argument(
        "--model",
        default="medmamba3d_tiny",
        choices=MODEL_CHOICES,
        help="Backbone used for evaluation.",
    )
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=12, help="Dataloader workers.")
    parser.add_argument("--device", default="cuda", help="Device string.")
    parser.add_argument("--test-runs", type=int, default=5, help="Number of repeated runs.")
    parser.add_argument("--swanlab-project", default="MedMamba", help="SwanLab project name.")
    parser.add_argument("--disable-swanlab", action="store_true", help="Disable SwanLab logging.")
    return parser.parse_args()


def resolve_checkpoint_path(checkpoint_arg, model_name):
    if checkpoint_arg:
        return Path(checkpoint_arg)
    return Path("assets/checkpoints_3d") / f"best_model_{model_name}.pth"


def main():
    args = parse_args()
    if swanlab is None:
        args.disable_swanlab = True

    print("\n" + "=" * 60)
    print(f"{'Model Testing Pipeline':^60}")
    print("=" * 60)

    start_time = time.time()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.model)
    test_dataset = Medical_Dataset(mode="test", csv_path=args.test_csv)
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "pin_memory": True,
        "num_workers": args.num_workers,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 8
    test_dataloader = DataLoader(test_dataset, **dataloader_kwargs)

    device_name = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    device = torch.device(device_name)
    model = build_model(args.model, args.num_classes)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    if not args.disable_swanlab:
        swanlab.init(
            project=args.swanlab_project,
            config={
                "phase": "test",
                "runs": args.test_runs,
                "batch_size": args.batch_size,
                "model_tag": args.model,
                "checkpoint": str(checkpoint_path),
                "test_csv": args.test_csv,
            },
        )

    metrics_list = []
    show_details = True
    for run_idx in range(args.test_runs):
        print(f"\n[Run {run_idx + 1}/{args.test_runs}]")
        metrics = evaluate_model(
            model,
            test_dataloader,
            device,
            verbose=True,
            show_details=show_details,
            desc=f"Test {run_idx + 1}",
        )
        metrics_list.append(metrics)
        if not args.disable_swanlab:
            swanlab.log(
                {
                    "run": run_idx + 1,
                    "accuracy": metrics[0],
                    "auc": metrics[1],
                    "sensitivity": metrics[2],
                    "specificity": metrics[3],
                    "f1": metrics[4],
                    "precision": metrics[5],
                    "mcc": metrics[6],
                }
            )
        show_details = False

    if args.test_runs > 1:
        metrics_arr = np.array(metrics_list, dtype=np.float32)
        mean = metrics_arr.mean(axis=0)
        std = metrics_arr.std(axis=0)
        print("\nMEAN ± STD")
        print(f"Accuracy: {mean[0]:.4f} ± {std[0]:.4f}")
        print(f"AUC:      {mean[1]:.4f} ± {std[1]:.4f}")
        print(f"Sens:     {mean[2]:.4f} ± {std[2]:.4f}")
        print(f"Spec:     {mean[3]:.4f} ± {std[3]:.4f}")
        print(f"F1:       {mean[4]:.4f} ± {std[4]:.4f}")
        print(f"Prec:     {mean[5]:.4f} ± {std[5]:.4f}")
        print(f"MCC:      {mean[6]:.4f} ± {std[6]:.4f}")
        if not args.disable_swanlab:
            swanlab.log(
                {
                    "mean/accuracy": mean[0],
                    "mean/auc": mean[1],
                    "mean/sensitivity": mean[2],
                    "mean/specificity": mean[3],
                    "mean/f1": mean[4],
                    "mean/precision": mean[5],
                    "mean/mcc": mean[6],
                    "std/accuracy": std[0],
                    "std/auc": std[1],
                    "std/sensitivity": std[2],
                    "std/specificity": std[3],
                    "std/f1": std[4],
                    "std/precision": std[5],
                    "std/mcc": std[6],
                }
            )

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Testing completed in {str(timedelta(seconds=int(total_time)))}")
    print("=" * 60 + "\n")

    if not args.disable_swanlab:
        swanlab.finish()


if __name__ == "__main__":
    main()
