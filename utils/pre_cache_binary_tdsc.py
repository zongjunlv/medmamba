import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from monai.transforms import Compose, EnsureChannelFirst, Resize, ScaleIntensity
from tqdm import tqdm

tfm = Compose([EnsureChannelFirst(channel_dim="no_channel"), ScaleIntensity(), Resize((256, 256, 128))])

LABEL_MAP = {"B": 0, "M": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached npy files for TDSC-like datasets.")
    parser.add_argument("--root", required=True, help="Dataset root containing labels_*.csv and split folders.")
    return parser.parse_args()


def build_split_configs(root: Path):
    return {
        "train": {
            "csv_path": root / "labels_train.csv",
            "cache_dir": root / "train_cache_dir",
            "cache_csv": root / "labels_train_cache.csv",
            "expected_dir": root / "train" / "imagesTr_origin",
        },
        "val": {
            "csv_path": root / "labels_val.csv",
            "cache_dir": root / "val_cache_dir",
            "cache_csv": root / "labels_val_cache.csv",
            "expected_dir": root / "val" / "imagesVal_origin",
        },
        "test": {
            "csv_path": root / "labels_test.csv",
            "cache_dir": root / "test_cache_dir",
            "cache_csv": root / "labels_test_cache.csv",
            "expected_dir": root / "test" / "imagesTs_origin",
        },
    }


def normalize_path(path_str: str) -> str:
    return os.path.normpath(path_str.replace("\\", os.sep))


def clear_cache_dir(cache_dir: Path) -> None:
    abs_dir = cache_dir.resolve()
    if not abs_dir.name.endswith("cache_dir"):
        raise ValueError(f"refusing to clear non-cache directory: {abs_dir}")
    if not abs_dir.exists():
        return

    for item in abs_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def load_sitk_image(data_path: Path) -> sitk.Image:
    if data_path.is_dir():
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(data_path))
        if not series_ids:
            raise ValueError(f"未在目录中找到 DICOM 序列: {data_path}")
        file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(data_path), series_ids[0])
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(file_names)
        return reader.Execute()
    return sitk.ReadImage(str(data_path))


def load_volume(data_path: Path) -> np.ndarray:
    image = load_sitk_image(data_path)
    volume_zyx = sitk.GetArrayFromImage(image).astype(np.float32)
    volume_xyz = np.transpose(volume_zyx, (2, 1, 0))
    return np.asarray(tfm(volume_xyz), dtype=np.float32)


def build_cache(split_name: str, csv_path: Path, cache_dir: Path, cache_csv: Path, expected_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    clear_cache_dir(cache_dir)

    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["case_id", "data_path"]).reset_index(drop=True)

    num_digits = max(4, len(str(len(df))))
    cached_rows = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"cache {split_name}", unit="item"):
        raw_data_path = Path(normalize_path(str(row["data_path"])))
        if raw_data_path.parent != expected_dir:
            raise ValueError(f"{split_name} split 路径不在预期目录下: {raw_data_path}")
        if not raw_data_path.exists():
            raise FileNotFoundError(f"找不到数据文件: {raw_data_path}")

        raw_label = str(row["label"]).strip().upper()
        if raw_label not in LABEL_MAP:
            raise ValueError(f"不支持的标签: {raw_label}")

        np_data = load_volume(raw_data_path)
        npy_name = f"{index + 1:0{num_digits}d}.npy"
        npy_path = cache_dir / npy_name
        np.save(npy_path, np_data)

        cached_rows.append(
            {
                "label": LABEL_MAP[raw_label],
                "data_path": str(raw_data_path),
                "npy_path": str(npy_path),
            }
        )

    pd.DataFrame(cached_rows).to_csv(cache_csv, index=False)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    for split_name, config in build_split_configs(root).items():
        build_cache(
            split_name=split_name,
            csv_path=config["csv_path"],
            cache_dir=config["cache_dir"],
            cache_csv=config["cache_csv"],
            expected_dir=config["expected_dir"],
        )


if __name__ == "__main__":
    main()
