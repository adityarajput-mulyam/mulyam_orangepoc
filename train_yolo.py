"""
Script to train / fine-tune YOLO model on tree crown dataset.
Supports YOLOv8 / YOLOv11 nano/small/medium models.
"""
import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

def train(
    data_yaml: str = "datasets/neon_trees/data.yaml",
    model_variant: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str = "runs/tree_crown_detection",
    name: str = "yolov8_citrus_poc"
):
    print("=" * 60)
    print(f"Starting YOLO GPU Training on {data_yaml}")
    print(f"Device: {device} (RTX 3050) | Epochs: {epochs} | Image Size: {imgsz} | Batch: {batch}")
    print("=" * 60)
    
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
    model = YOLO(model_variant)
    
    results = model.train(
        data=str(yaml_path.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=15,
        project=project,
        name=name,
        plots=True,
        save=True,
        verbose=True,
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,
        degrees=15.0,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4
    )
    
    print("\n" + "=" * 60)
    print(f"Training Complete! Results saved to: {results.save_dir}")
    print("=" * 60)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO for Tree Crown Detection")
    parser.add_argument("--data", type=str, default="datasets/neon_trees/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model checkpoint (e.g. yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    
    args = parser.parse_args()
    train(data_yaml=args.data, model_variant=args.model, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
