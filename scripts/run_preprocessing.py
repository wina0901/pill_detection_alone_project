import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing.split import split_coco_train_val
from src.preprocessing.augmentation import extract_minority_crops, run_copy_paste
from src.preprocessing.transforms import run_letterbox_pipeline, apply_clahe_to_folder
from src.preprocessing.format_converter import run_yolo_conversion


def main():
    BASE_DIR = "/content/drive/MyDrive/data/doit"

    # 원본 입력
    merged_train_json = os.path.join(BASE_DIR, "merged_annotations_train_final.json")
    train_img_dir = os.path.join(BASE_DIR, "train_images")

    # split 결과
    train_raw_json = os.path.join(BASE_DIR, "train_raw.json")
    val_raw_json = os.path.join(BASE_DIR, "val_raw.json")

    # 증강 결과
    train_aug_json = os.path.join(BASE_DIR, "train_augmented_final.json")

    # letterbox 결과
    train_letterbox_json = os.path.join(BASE_DIR, "train_letterbox.json")
    val_letterbox_json = os.path.join(BASE_DIR, "val_letterbox.json")
    train_letterbox_dir = os.path.join(BASE_DIR, "letterbox_images", "train")
    val_letterbox_dir = os.path.join(BASE_DIR, "letterbox_images", "val")

    print("\n" + "=" * 70)
    print("STEP 1. Train / Val Split")
    print("=" * 70)
    split_coco_train_val(
        train_json_path=merged_train_json,
        train_save_path=train_raw_json,
        val_save_path=val_raw_json,
        val_ratio=0.2,
        seed=42,
    )

    print("\n" + "=" * 70)
    print("STEP 2. Minority Crop Extraction")
    print("=" * 70)
    extract_minority_crops(
        base_dir=BASE_DIR,
        threshold=50,
    )

    print("\n" + "=" * 70)
    print("STEP 3. Copy-Paste Augmentation (train only)")
    print("=" * 70)
    run_copy_paste(
        base_dir=BASE_DIR,
        aug_count=300,   # 처음엔 300 정도로 시작 추천
        random_seed=42,
    )

    print("\n" + "=" * 70)
    print("STEP 4. Letterbox 800x800")
    print("=" * 70)
    run_letterbox_pipeline(
        json_path=train_aug_json,
        out_json_path=train_letterbox_json,
        img_out_dir=train_letterbox_dir,
        base_dir=BASE_DIR,
        target_size=800,
        desc="Train Letterbox",
    )
    run_letterbox_pipeline(
        json_path=val_raw_json,
        out_json_path=val_letterbox_json,
        img_out_dir=val_letterbox_dir,
        base_dir=BASE_DIR,
        target_size=800,
        desc="Val Letterbox",
    )

    print("\n" + "=" * 70)
    print("STEP 5. CLAHE")
    print("=" * 70)
    apply_clahe_to_folder(train_letterbox_dir, clip_limit=2.0, tile_grid_size=(8, 8))
    apply_clahe_to_folder(val_letterbox_dir, clip_limit=2.0, tile_grid_size=(8, 8))

    print("\n" + "=" * 70)
    print("STEP 6. YOLO / RT-DETR Format Conversion")
    print("=" * 70)
    run_yolo_conversion(BASE_DIR)

    print("\n✅ 전처리 전체 완료")


if __name__ == "__main__":
    main()