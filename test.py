import argparse
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
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
    parser.add_argument(
        "--export-pred-csv",
        default=None,
        help="Optional path to export per-sample predictions. Defaults to assets/test_outputs/predictions_<model>.csv.",
    )
    parser.add_argument("--swanlab-project", default="MedMamba", help="SwanLab project name.")
    parser.add_argument("--disable-swanlab", action="store_true", help="Disable SwanLab logging.")
    return parser.parse_args()


def resolve_checkpoint_path(checkpoint_arg, model_name):
    if checkpoint_arg:
        return Path(checkpoint_arg)
    return Path("assets/checkpoints_3d") / f"best_model_{model_name}.pth"


def resolve_export_path(export_arg, model_name):
    if export_arg:
        return Path(export_arg)
    return Path("assets/test_outputs") / f"predictions_{model_name}.csv"


def print_probability_summary(labels_arr, probs_arr):
    pos_probs = probs_arr[:, 1] if probs_arr.shape[1] > 1 else probs_arr[:, 0]
    print("\nPositive-Class Probability Summary")
    print(f"all   : min={pos_probs.min():.4f} q25={np.quantile(pos_probs, 0.25):.4f} "
          f"median={np.quantile(pos_probs, 0.5):.4f} q75={np.quantile(pos_probs, 0.75):.4f} max={pos_probs.max():.4f}")
    for label in sorted(np.unique(labels_arr).tolist()):
        mask = labels_arr == label
        cls_probs = pos_probs[mask]
        if len(cls_probs) == 0:
            continue
        print(f"label={label}: min={cls_probs.min():.4f} q25={np.quantile(cls_probs, 0.25):.4f} "
              f"median={np.quantile(cls_probs, 0.5):.4f} q75={np.quantile(cls_probs, 0.75):.4f} max={cls_probs.max():.4f}")


def export_predictions_csv(export_path, dataset, labels_arr, preds_arr, probs_arr):
    export_path.parent.mkdir(parents=True, exist_ok=True)
    df = dataset.data.copy().reset_index(drop=True)
    df["label_true"] = labels_arr.astype(int)
    df["label_pred"] = preds_arr.astype(int)
    if probs_arr.shape[1] == 1:
        df["prob_0"] = probs_arr[:, 0]
    else:
        for idx in range(probs_arr.shape[1]):
            df[f"prob_{idx}"] = probs_arr[:, idx]
    df.to_csv(export_path, index=False)
    print(f"\nPer-sample predictions exported to: {export_path}")


def main():
    args = parse_args()
    if swanlab is None:
        args.disable_swanlab = True

    print("\n" + "=" * 60)
    print(f"{'Model Testing Pipeline':^60}")
    print("=" * 60)

    start_time = time.time()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.model)
    export_path = resolve_export_path(args.export_pred_csv, args.model)
    test_dataset = Medical_Dataset(mode="test", csv_path=args.test_csv, roi_size=(256, 256, 128), margin=12)
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
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    if not args.disable_swanlab:
        swanlab.init(
            project=args.swanlab_project,
            config={
                "phase": "test",
                "batch_size": args.batch_size,
                "model_tag": args.model,
                "checkpoint": str(checkpoint_path),
                "test_csv": args.test_csv,
            },
        )

    metrics = evaluate_model(
        model,
        test_dataloader,
        device,
        verbose=True,
        show_details=True,
        desc="Test",
        return_outputs=True,
    )
    accuracy, auc, sensitivity, specificity, f1, precision, mcc, labels_arr, preds_arr, probs_arr = metrics
    print_probability_summary(labels_arr, probs_arr)
    export_predictions_csv(export_path, test_dataset, labels_arr, preds_arr, probs_arr)

    if not args.disable_swanlab:
        swanlab.log(
            {
                "accuracy": accuracy,
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
    print(f"Testing completed in {str(timedelta(seconds=int(total_time)))}")
    print("=" * 60 + "\n")

    if not args.disable_swanlab:
        swanlab.finish()


if __name__ == "__main__":
    main()
