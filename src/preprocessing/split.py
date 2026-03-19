import json
import random
from pathlib import Path
from collections import defaultdict


def load_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, save_path: str):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def split_coco_train_val(
    train_json_path: str,
    train_save_path: str,
    val_save_path: str,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """
    merged_annotations_train_final.json -> train_raw.json / val_raw.json
    image 단위로 split하고, annotation은 image_id 기준으로 따라 붙입니다.
    """
    coco = load_json(train_json_path)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    random.seed(seed)
    images_shuffled = images[:]
    random.shuffle(images_shuffled)

    n_val = int(len(images_shuffled) * val_ratio)
    val_images = images_shuffled[:n_val]
    train_images = images_shuffled[n_val:]

    train_image_ids = {img["id"] for img in train_images}
    val_image_ids = {img["id"] for img in val_images}

    train_annotations = [ann for ann in annotations if ann["image_id"] in train_image_ids]
    val_annotations = [ann for ann in annotations if ann["image_id"] in val_image_ids]

    train_coco = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories,
    }
    val_coco = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": categories,
    }

    save_json(train_coco, train_save_path)
    save_json(val_coco, val_save_path)

    print(f"✅ train_raw.json 저장: {train_save_path}")
    print(f"   이미지 {len(train_images)}장 / 객체 {len(train_annotations)}개")
    print(f"✅ val_raw.json 저장: {val_save_path}")
    print(f"   이미지 {len(val_images)}장 / 객체 {len(val_annotations)}개")