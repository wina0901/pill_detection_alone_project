import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.category_mapping import (
    build_category_mappings,
    save_category_mappings,
    print_mapping_summary,
)


if __name__ == "__main__":
    BASE_DIR = "/content/drive/MyDrive/data/doit"

    train_json_path = f"{BASE_DIR}/merged_annotations_train_final.json"
    save_path = "artifacts/category_mapping.json"

    mapping = build_category_mappings(train_json_path)
    print_mapping_summary(mapping)
    save_category_mappings(mapping, save_path)