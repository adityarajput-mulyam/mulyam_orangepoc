"""
Optimized YOLO training for citrus tree crown counting.
Targets 80-88% mAP50 without overfitting.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ultralytics import YOLO
import yaml

DATA_YAML = ROOT_DIR / "datasets/neon_trees/data.yaml"
PROJECT = str(ROOT_DIR / "runs/detect/runs/tree_crown_detection")
RUN_NAME = "citrus_optimized_v4"

def patch_data_yaml():
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str(ROOT_DIR / "datasets/neon_trees")
    with open(DATA_YAML, "w") as f:
        yaml.dump(cfg, f)
    print(f"data.yaml path set to: {cfg['path']}")

def train():
    patch_data_yaml()

    model = YOLO("yolov8s.pt")

    print("=" * 60)
    print(f"Training: {RUN_NAME}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Target: 80-88% mAP50 for tree crown counting")
    print("=" * 60)

    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({'GPU: ' + torch.cuda.get_device_name(0) if device == '0' else 'CPU'})")

    results = model.train(
        data=str(DATA_YAML),
        epochs=150,
        imgsz=640,
        batch=8,
        device=device,
        patience=30,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,           # Overwrite existing run dir
        # --- Optimizer: SGD works better than AdamW for YOLO on small datasets ---
        optimizer="SGD",
        lr0=0.01,                # Standard YOLO LR
        lrf=0.1,                 # Final LR = lr0 * lrf = 0.001 (healthy range)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        cos_lr=True,
        # --- Loss ---
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # --- Augmentation ---
        degrees=45.0,            # Reduced rotation (90 caused instability)
        flipud=0.5,
        fliplr=0.5,
        scale=0.5,
        mosaic=1.0,              # Full mosaic
        close_mosaic=30,         # Keep mosaic until epoch 120 (was default 10)
        mixup=0.1,               # Light mixup OK with SGD
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        shear=0.0,
        perspective=0.0,
        # --- IoU / NMS ---
        iou=0.5,                 # Standard IoU threshold
        conf=0.001,              # Very low conf during training for recall
        # --- Output ---
        plots=True,
        save=True,
        save_period=20,
        workers=0,               # CRITICAL: 0 = no subprocess workers (WinError 1455 fix)
    )

    print("\n" + "=" * 60)
    print(f"Training complete! Results: {results.save_dir}")
    best_map = results.results_dict.get("metrics/mAP50(B)", "N/A")
    print(f"Best mAP50: {best_map}")
    print("=" * 60)
    return results

if __name__ == "__main__":
    train()
