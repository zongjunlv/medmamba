import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from monai.transforms import Compose, EnsureChannelFirst, Resize, ScaleIntensity
from tqdm import tqdm

IMAGE_TFM = Compose([EnsureChannelFirst(channel_dim="no_channel"), ScaleIntensity(), Resize((256, 256, 128))])
MASK_TFM = Compose([EnsureChannelFirst(channel_dim="no_channel"), Resize((256, 256, 128), mode="nearest")])
DEFAULT_MARGIN = 12

LABEL_MAP = {"B": 0, "M": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached npy files for TDSC-like datasets.")
    parser.add_argument("--root", required=True, help="Dataset root containing labels_*.csv and split folders.")
    parser.add_argument(
        "--target",
        choices=["image", "mask", "both"],
        default="both",
        help="Choose whether to process image cache, mask cache, or both.",
    )
    return parser.parse_args()


def build_split_configs(root: Path):
    return {
        "train": {
            "csv_path": root / "labels_train.csv",
            "image_cache_dir": root / "train_cache_dir",
            "mask_cache_dir": root / "train_mask_cache_dir",
            "cache_csv": root / "labels_train_cache.csv",
            "expected_image_dir": root / "train" / "imagesTr_origin",
            "expected_mask_dir": root / "train" / "labelsTr_origin",
        },
        "val": {
            "csv_path": root / "labels_val.csv",
            "image_cache_dir": root / "val_cache_dir",
            "mask_cache_dir": root / "val_mask_cache_dir",
            "cache_csv": root / "labels_val_cache.csv",
            "expected_image_dir": root / "val" / "imagesVal_origin",
            "expected_mask_dir": root / "val" / "labelsVal_origin",
        },
        "test": {
            "csv_path": root / "labels_test.csv",
            "image_cache_dir": root / "test_cache_dir",
            "mask_cache_dir": root / "test_mask_cache_dir",
            "cache_csv": root / "labels_test_cache.csv",
            "expected_image_dir": root / "test" / "imagesTs_origin",
            "expected_mask_dir": root / "test" / "labelsTs_origin",
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


def load_raw_volume(data_path: Path) -> np.ndarray:
    image = load_sitk_image(data_path)
    volume_zyx = sitk.GetArrayFromImage(image).astype(np.float32)
    volume_xyz = np.transpose(volume_zyx, (2, 1, 0))
    return np.asarray(volume_xyz, dtype=np.float32)


def crop_by_mask(image_xyz: np.ndarray, mask_xyz: np.ndarray, margin: int = DEFAULT_MARGIN) -> tuple[np.ndarray, np.ndarray]:
    mask_fg = mask_xyz > 0
    if mask_fg.sum() == 0:
        return image_xyz, mask_xyz

    coords = np.argwhere(mask_fg)
    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0) + 1

    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    z_min = max(0, z_min - margin)
    x_max = min(image_xyz.shape[0], x_max + margin)
    y_max = min(image_xyz.shape[1], y_max + margin)
    z_max = min(image_xyz.shape[2], z_max + margin)

    image_crop = image_xyz[x_min:x_max, y_min:y_max, z_min:z_max]
    mask_crop = mask_xyz[x_min:x_max, y_min:y_max, z_min:z_max]
    return image_crop, mask_crop


def finalize_image(volume_xyz: np.ndarray) -> np.ndarray:
    return np.asarray(IMAGE_TFM(volume_xyz), dtype=np.float32)


def finalize_mask(volume_xyz: np.ndarray) -> np.ndarray:
    return np.asarray(MASK_TFM(volume_xyz), dtype=np.float32)


def load_existing_cache(cache_csv: Path) -> dict[str, dict]:
    if not cache_csv.exists():
        return {}

    df = pd.read_csv(cache_csv)
    if "data_path" not in df.columns:
        return {}
    records = {}
    for _, row in df.iterrows():
        records[str(row["data_path"])] = row.to_dict()
    return records


def build_cache(
    split_name: str,
    csv_path: Path,
    image_cache_dir: Path,
    mask_cache_dir: Path,
    cache_csv: Path,
    expected_image_dir: Path,
    expected_mask_dir: Path,
    target: str,
) -> None:
    process_image = target in {"image", "both"}
    process_mask = target in {"mask", "both"}

    if process_image:
        image_cache_dir.mkdir(parents=True, exist_ok=True)
        clear_cache_dir(image_cache_dir)
    if process_mask:
        mask_cache_dir.mkdir(parents=True, exist_ok=True)
        clear_cache_dir(mask_cache_dir)

    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["case_id", "data_path"]).reset_index(drop=True)
    existing_cache = load_existing_cache(cache_csv)

    num_digits = max(4, len(str(len(df))))
    cached_rows = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"cache {split_name}", unit="item"):
        raw_data_path = Path(normalize_path(str(row["data_path"])))
        raw_mask_path = Path(normalize_path(str(row["mask_path"])))
        if raw_data_path.parent != expected_image_dir:
            raise ValueError(f"{split_name} split image 路径不在预期目录下: {raw_data_path}")
        if raw_mask_path.parent != expected_mask_dir:
            raise ValueError(f"{split_name} split mask 路径不在预期目录下: {raw_mask_path}")
        if not raw_data_path.exists():
            raise FileNotFoundError(f"找不到 image 文件: {raw_data_path}")
        if not raw_mask_path.exists():
            raise FileNotFoundError(f"找不到 mask 文件: {raw_mask_path}")

        raw_label = str(row["label"]).strip().upper()
        if raw_label not in LABEL_MAP:
            raise ValueError(f"不支持的标签: {raw_label}")

        record_key = str(raw_data_path)
        existing_row = existing_cache.get(record_key, {})
        npy_name = f"{index + 1:0{num_digits}d}.npy"
        raw_image_xyz = None
        raw_mask_xyz = None
        if process_image or process_mask:
            raw_image_xyz = load_raw_volume(raw_data_path)
            raw_mask_xyz = load_raw_volume(raw_mask_path)
            raw_image_xyz, raw_mask_xyz = crop_by_mask(raw_image_xyz, raw_mask_xyz, margin=DEFAULT_MARGIN)

        if process_image:
            np_data = finalize_image(raw_image_xyz)
            image_npy_path = image_cache_dir / npy_name
            np.save(image_npy_path, np_data)
        else:
            image_npy_path = existing_row.get("npy_path", "")

        if process_mask:
            np_mask = finalize_mask(raw_mask_xyz)
            mask_npy_path = mask_cache_dir / npy_name
            np.save(mask_npy_path, np_mask)
        else:
            mask_npy_path = existing_row.get("mask_path", "")

        cached_rows.append(
            {
                "label": LABEL_MAP[raw_label],
                "data_path": str(raw_data_path),
                "npy_path": str(image_npy_path),
                "mask_path": str(mask_npy_path),
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
            image_cache_dir=config["image_cache_dir"],
            mask_cache_dir=config["mask_cache_dir"],
            cache_csv=config["cache_csv"],
            expected_image_dir=config["expected_image_dir"],
            expected_mask_dir=config["expected_mask_dir"],
            target=args.target,
        )


if __name__ == "__main__":
    main()
