"""
Benchmark and compare YOLO models on NEON Tree Crown dataset:
- Standard detection metrics (mAP50, mAP50-95, P, R)
- Direct Tree Counting Accuracy (MAE, MAPE, Count Accuracy = 1 - MAPE)
"""
import os
import glob
from pathlib import Path
import numpy as np
from ultralytics import YOLO

DATA_YAML = "datasets/neon_trees/data.yaml"
MODELS = {
    "yolov8n_citrus_poc2 (74 epochs)": "runs/detect/runs/tree_crown_detection/yolov8_citrus_poc-2/weights/best.pt",
    "yolov8n_citrus_optimized (35 epochs)": "runs/detect/runs/tree_crown_detection/yolov8_citrus_optimized/weights/best.pt",
}

def count_ground_truth(label_dir):
    gt_counts = {}
    for p in Path(label_dir).glob("*.txt"):
        with open(p, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        gt_counts[p.stem] = len(lines)
    return gt_counts

def evaluate():
    val_labels_dir = "datasets/neon_trees/val/labels"
    val_images_dir = "datasets/neon_trees/val/images"
    gt_counts = count_ground_truth(val_labels_dir)
    total_gt = sum(gt_counts.values())

    print("=" * 70)
    print(f"EVALUATING ON VALIDATION SET: {len(gt_counts)} images, {total_gt} total tree crowns")
    print("=" * 70)

    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"Skipping {name}, not found at {path}")
            continue

        try:
            model = YOLO(path)
        except Exception as e:
            print(f"Skipping {name}: not a standard Ultralytics YOLO checkpoint ({e})")
            continue

        # 1. Standard YOLO val
        val_res = model.val(data=DATA_YAML, split="val", imgsz=640, device="0", verbose=False)
        mAP50 = val_res.box.map50
        mAP50_95 = val_res.box.map
        prec = val_res.box.mp
        rec = val_res.box.mr

        print(f"Detection Results: mAP@50={mAP50*100:.2f}%, mAP@50-95={mAP50_95*100:.2f}%, Precision={prec*100:.2f}%, Recall={rec*100:.2f}%")

        # 2. Count Accuracy across confidence thresholds
        for conf in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            results = model.predict(source=val_images_dir, conf=conf, imgsz=640, device="0", verbose=False)
            pred_counts = {Path(r.path).stem: len(r.boxes) for r in results}

            errors = []
            total_pred = 0
            for stem, gt_c in gt_counts.items():
                p_c = pred_counts.get(stem, 0)
                total_pred += p_c
                if gt_c > 0:
                    err = abs(p_c - gt_c) / gt_c
                    errors.append(err)

            mape = np.mean(errors) if errors else 1.0
            count_acc = max(0.0, (1.0 - mape)) * 100.0
            overall_ratio_acc = (1.0 - abs(total_pred - total_gt) / total_gt) * 100.0 if total_gt else 0.0

            print(f"  [Conf {conf:.2f}] Total Pred={total_pred}/{total_gt} | Mean Image Count Acc={count_acc:.2f}% | Fleet Count Acc={overall_ratio_acc:.2f}%")

if __name__ == "__main__":
    evaluate()
