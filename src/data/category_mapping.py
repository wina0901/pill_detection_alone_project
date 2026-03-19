import json
from pathlib import Path


def load_json(json_path: str):
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def build_category_mappings(train_json_path: str) -> dict:
    data = load_json(train_json_path)
    categories = data.get("categories", [])

    if not categories:
        raise ValueError("No categories found in train json.")

    orig2model = {}
    model2orig = {}
    category_info = []

    for model_id, cat in enumerate(categories):
        orig_id = int(cat["id"])
        cat_name = cat.get("name", f"class_{model_id}")

        orig2model[str(orig_id)] = model_id
        model2orig[str(model_id)] = orig_id

        category_info.append({
            "model_id": model_id,
            "orig_id": orig_id,
            "name": cat_name,
        })

    mapping = {
        "num_classes": len(categories),
        "orig2model": orig2model,
        "model2orig": model2orig,
        "categories": category_info,
    }

    return mapping


def save_category_mappings(mapping: dict, save_path: str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"[INFO] category mapping saved to: {save_path}")


def print_mapping_summary(mapping: dict) -> None:
    print("=" * 60)
    print("Category Mapping Summary")
    print(f"Number of classes: {mapping['num_classes']}")
    print("-" * 60)

    print("First 10 mappings (orig_id -> model_id):")
    items = list(mapping["orig2model"].items())[:10]
    for orig_id, model_id in items:
        print(f"{orig_id} -> {model_id}")

    print("-" * 60)
    print("First 10 category infos:")
    for item in mapping["categories"][:10]:
        print(item)

    print("=" * 60)