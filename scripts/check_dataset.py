import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.inspect_json import summarize_coco_json, print_summary, compare_train_test_categories


if __name__ == "__main__":
    BASE_DIR = "/content/drive/MyDrive/data/doit"

    train_json_path = f"{BASE_DIR}/merged_annotations_train_final.json"
    test_json_path = f"{BASE_DIR}/merged_annotations_test_final.json"

    train_summary = summarize_coco_json(train_json_path)
    test_summary = summarize_coco_json(test_json_path)

    print_summary(train_summary)
    print_summary(test_summary)

    compare_train_test_categories(train_json_path, test_json_path)