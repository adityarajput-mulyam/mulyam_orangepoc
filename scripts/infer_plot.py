"""
Inference script to detect and count trees in aerial/satellite imagery,
filter detections by user plot polygon, and calculate canopy metrics.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from shapely.geometry import Polygon, Point

def count_trees_in_plot(
    image_path: str,
    model_path: str = "yolov8n.pt",
    polygon_points: list = None,
    conf_threshold: float = 0.25,
    save_output: str = "output_detection.jpg"
):
    """
    Detects tree crowns in the given image.
    If polygon_points is provided (list of [x, y] in image coords or normalized),
    filters only trees inside the polygon and returns the count.
    """
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at {image_path}")
        
    h, w, _ = img.shape
    
    # Load YOLO model
    model = YOLO(model_path)
    results = model.predict(source=img, conf=conf_threshold, verbose=False)[0]
    
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    
    # Setup polygon mask if provided
    poly = None
    if polygon_points and len(polygon_points) >= 3:
        # Check if coordinates are normalized (0 to 1) or pixel
        pts = np.array(polygon_points, dtype=np.float32)
        if pts.max() <= 1.0:
            pts[:, 0] *= w
            pts[:, 1] *= h
        poly = Polygon(pts)
        
        # Draw user plot boundary in Cyan
        pts_int = pts.astype(np.int32)
        cv2.polylines(img, [pts_int], isClosed=True, color=(255, 255, 0), thickness=3)
        
    tree_count = 0
    total_detected = len(boxes)
    
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        
        inside = True
        if poly is not None:
            inside = poly.contains(Point(cx, cy))
            
        if inside:
            tree_count += 1
            # Draw green bounding box for trees inside user plot
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(img, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        else:
            # Draw faded gray box for trees outside plot
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (128, 128, 128), 1)
            
    # Add count banner
    banner_text = f"Citrus Trees in Plot: {tree_count}" if poly else f"Total Trees Detected: {tree_count}"
    cv2.rectangle(img, (10, 10), (380, 55), (0, 0, 0), -1)
    cv2.putText(img, banner_text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    cv2.imwrite(save_output, img)
    print(f"\n[RESULTS]")
    print(f"Total Detections in Image: {total_detected}")
    print(f"Trees Inside Defined Plot: {tree_count}")
    print(f"Annotated Image Saved: {save_output}")
    
    return {
        "total_detected": total_detected,
        "plot_tree_count": tree_count,
        "output_path": save_output
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img_arg = sys.argv[1]
        model_arg = sys.argv[2] if len(sys.argv) > 2 else "yolov8n.pt"
        count_trees_in_plot(img_arg, model_arg)
    else:
        print("Usage: python infer_plot.py <image_path> [model_path]")
