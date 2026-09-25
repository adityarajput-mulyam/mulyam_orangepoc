"""
Evaluate counting accuracy of the YOLO model on the test split.
Reports:
  - True Count (annotations)
  - Predicted Count (detections at given conf)
  - Count Error % per image
  - Mean Absolute Error, Mean Absolute Percentage Error
  - Images within +-20% of true count (target: 80%+)
"""
import sys
import csv
import math
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ultralytics import YOLO

def evaluate_counting(model_path: str, data_yaml: str, conf: float = 0.25, iou: float = 0.4):
    model = YOLO(model_path)

    # Run validation to get mAP
    val_results = model.val(
        data=data_yaml,
        conf=conf,
        iou=iou,
        split="test",
        device="cpu",
        verbose=False,
    )

    map50 = val_results.results_dict.get("metrics/mAP50(B)", 0)
    map50_95 = val_results.results_dict.get("metrics/mAP50-95(B)", 0)
    precision = val_results.results_dict.get("metrics/precision(B)", 0)
    recall = val_results.results_dict.get("metrics/recall(B)", 0)

    # Per-image counting accuracy
    test_img_dir = Path(data_yaml).parent / "test" / "images"
    test_lbl_dir = Path(data_yaml).parent / "test" / "labels"

    errors = []
    within_20pct = 0
    total = 0

    for img_path in sorted(test_img_dir.glob("*.jpg")):
        lbl_path = test_lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        # True count from label file
        true_count = sum(1 for l in open(lbl_path) if l.strip())

        # Predicted count from model
        preds = model.predict(str(img_path), conf=conf, iou=iou, verbose=False, device="cpu")
        pred_count = len(preds[0].boxes) if preds and preds[0].boxes else 0

        if true_count == 0:
            continue

        pct_err = abs(pred_count - true_count) / true_count * 100
        errors.append(pct_err)
        if pct_err <= 20.0:
            within_20pct += 1
        total += 1

    mae = sum(errors) / len(errors) if errors else 0
    within_20pct_pct = within_20pct / total * 100 if total else 0

    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Conf={conf}, IoU={iou}, Test images={total}")
    print("─" * 60)
    print(f"mAP50:              {map50:.4f}  ({map50*100:.1f}%)")
    print(f"mAP50-95:           {map50_95:.4f}  ({map50_95*100:.1f}%)")
    print(f"Precision:          {precision:.4f}")
    print(f"Recall:             {recall:.4f}")
    print("─" * 60)
    print(f"Mean Abs Count Err: {mae:.1f}%")
    print(f"Within ±20% count:  {within_20pct}/{total}  ({within_20pct_pct:.1f}%)")
    print("=" * 60)

    return {
        "mAP50": map50,
        "mAP50_95": map50_95,
        "precision": precision,
        "recall": recall,
        "mean_count_err_pct": mae,
        "within_20pct": within_20pct_pct,
    }

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/detect/runs/tree_crown_detection/citrus_optimized_v3/weights/best.pt")
    p.add_argument("--data", default="datasets/neon_trees/data.yaml")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.4)
    args = p.parse_args()
    evaluate_counting(args.model, args.data, args.conf, args.iou)
