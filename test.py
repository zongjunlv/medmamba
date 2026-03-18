import os
import time
from datetime import timedelta

import torch
import numpy as np
import swanlab
from torch.utils.data import DataLoader

from data.medical_dataset import Medical_Dataset
from models import ResNet3D, R2Plus1DClassifier, R3DClassifier, UNetEncoderClassifier, SwinClassifier3D
from utils.evaluator import evaluate_model


def main():
    print("\n" + "=" * 60)
    print(f"{'Model Testing Pipeline':^60}")
    print("=" * 60)

    start_time = time.time()

    test_path = '/data02/workspace/LZJ_SPACE/MedMamba/dataset/ABUS_Classification/330_512_512/330_512_512_test.csv'
    # test_path = '/data02/workspace/LZJ_SPACE/MedMamba/dataset/ABUS_Classification/ABUS_NEW/split_test_7_1_2_cache.csv'

    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    test_runs = int(os.getenv("TEST_RUNS", "5"))

    test_dataset = Medical_Dataset(mode='test', csv_path=test_path)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=12,
        persistent_workers=True,
        prefetch_factor=8,
    )

    # 1) MedicalNet 预训练 3D ResNet
    model = ResNet3D(
        variant="resnet34",
        num_classes=3,
        in_chans=1,
        pretrained=True
    )

    # 2) R(2+1)D (Kinetics400 预训练，输入视作时间维 T)
    # model = R2Plus1DClassifier(
    #     num_classes=3,
    #     in_chans=1,
    #     pretrained=True
    # )

    # 3) R3D/I3D (Kinetics400 预训练)
    # model = R3DClassifier(
    #     num_classes=3,
    #     in_chans=1,
    #     pretrained=True
    # )

    # 4) 3D UNet 编码器 + 分类头（无预训练）
    # model = UNetEncoderClassifier(
    #     num_classes=3,
    #     in_chans=1
    # )

    # 5) Swin 3D（可加载 MONAI SSL 预训练权重）
    # swin_kwargs = dict(num_classes=3, in_chans=1)
    # model = SwinClassifier3D(
    #     **swin_kwargs,
    #     pretrained=False,
    # )

    model_name = getattr(model, "variant", model.__class__.__name__)
    safe_model_name = str(model_name).replace(" ", "_")

    checkpoint_path = os.path.join('assets', 'checkpoints_3d', f'best_model_{safe_model_name}.pth')
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    swanlab.init(
        project="my-awesome-project",
        config={
            "phase": "test",
            "runs": test_runs,
            "batch_size": batch_size,
            "model_tag": model_name,
            "checkpoint": checkpoint_path,
        },
    )

    metrics_list = []
    show_details = True
    for run_idx in range(test_runs):
        print(f"\n[Run {run_idx + 1}/{test_runs}]")
        metrics = evaluate_model(
            model,
            test_dataloader,
            device,
            verbose=True,
            show_details=show_details,
            desc=f"Test {run_idx + 1}",
        )
        metrics_list.append(metrics)
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

    if test_runs > 1:
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
    swanlab.finish()


if __name__ == '__main__':
    main()
