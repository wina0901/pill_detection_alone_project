import json
from pathlib import Path


def load_json(json_path: str):
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def summarize_coco_json(json_path: str) -> dict:
    data = load_json(json_path)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    summary = {
        "json_path": str(json_path),
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_categories": len(categories),
        "sample_image": images[0] if images else None,
        "sample_annotation": annotations[0] if annotations else None,
        "sample_category": categories[0] if categories else None,
    }

    return summary


def print_summary(summary: dict) -> None:
    print("=" * 60)
    print(f"File: {summary['json_path']}")
    print(f"Number of images      : {summary['num_images']}")
    print(f"Number of annotations : {summary['num_annotations']}")
    print(f"Number of categories  : {summary['num_categories']}")
    print("-" * 60)
    print("Sample image:")
    print(summary["sample_image"])
    print("-" * 60)
    print("Sample annotation:")
    print(summary["sample_annotation"])
    print("-" * 60)
    print("Sample category:")
    print(summary["sample_category"])
    print("=" * 60)


def extract_category_ids(json_path: str):
    data = load_json(json_path)
    categories = data.get("categories", [])
    return [cat["id"] for cat in categories]


def compare_train_test_categories(train_json_path: str, test_json_path: str) -> None:
    train_ids = extract_category_ids(train_json_path)
    test_ids = extract_category_ids(test_json_path)

    print("\n[Category ID Check]")
    print(f"Train category count: {len(train_ids)}")
    print(f"Test category count : {len(test_ids)}")
    print(f"Same category ids   : {set(train_ids) == set(test_ids)}")

    print("\nFirst 10 train category ids:")
    print(train_ids[:10])

    print("\nFirst 10 test category ids:")
    print(test_ids[:10])