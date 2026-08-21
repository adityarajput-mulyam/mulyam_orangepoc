"""
Citrus Tree Counting & Farm Analytics API Server
"""
import os
import sys
import math
import base64
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from shapely.geometry import Polygon, Point

from services.satellite_service import SatelliteService
from services.analytics import FarmAnalyticsService

app = FastAPI(title="Orange Tree Counting & Orchard Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths
RUN_DIRS = sorted(Path("runs/detect/runs/tree_crown_detection").glob("yolov8_citrus_poc*"))
LATEST_MODEL = RUN_DIRS[-1] / "weights" / "best.pt" if RUN_DIRS else Path("models/NEON.pt")

active_model_path = str(LATEST_MODEL) if LATEST_MODEL.exists() else "yolov8n.pt"
print(f"Loading YOLO Tree Detection Model from: {active_model_path}")
yolo_model = YOLO(active_model_path)

satellite_service = SatelliteService(zoom=18)
analytics_service = FarmAnalyticsService(avg_yield_per_tree_kg=55.0, price_per_kg_inr=35.0)

class PlotAnalysisRequest(BaseModel):
    coordinates: List[List[float]] # [[lon, lat], [lon, lat], ...]
    conf_threshold: Optional[float] = 0.12
    farm_name: Optional[str] = "Nagpur Citrus Orchard"
    market_price_per_kg: Optional[float] = 35.0
    zoom_level: Optional[int] = 18
    provider: Optional[str] = "google"
    sensitivity: Optional[float] = 0.5

@app.post("/api/analyze-plot")
async def analyze_farm_plot(req: PlotAnalysisRequest):
    coords = req.coordinates
    if len(coords) < 3:
        raise HTTPException(status_code=400, detail="At least 3 coordinate points required to define a plot polygon.")

    # 1. Calculate farm area in hectares
    farm_area_ha = analytics_service.compute_polygon_area_ha(coords)

    # 2. Fetch high-res satellite imagery
    zoom = req.zoom_level or 18
    try:
        img_bgr, geo_meta = satellite_service.fetch_satellite_image_for_polygon(
            coords,
            zoom=zoom,
            provider=req.provider or "google"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Satellite imagery fetch error: {str(e)}")

    h, w, _ = img_bgr.shape
    poly_pixels = geo_meta["polygon_pixels"]
    poly_geom = Polygon(poly_pixels)

    # 3. Sliced Sliding-Window Inference (High-Res SAHI-style Tiling)
    patch_size = 512
    step = int(patch_size * 0.75)
    conf_thresh = float(req.conf_threshold) if req.conf_threshold else 0.12

    raw_boxes = []
    raw_confs = []

    y_steps = list(range(0, max(1, h - patch_size + 1), step))
    if y_steps[-1] + patch_size < h:
        y_steps.append(h - patch_size)

    x_steps = list(range(0, max(1, w - patch_size + 1), step))
    if x_steps[-1] + patch_size < w:
        x_steps.append(w - patch_size)

    for y in y_steps:
        for x in x_steps:
            patch = img_bgr[y:y+patch_size, x:x+patch_size]
            results = yolo_model.predict(source=patch, conf=conf_thresh, verbose=False)[0]
            for box, conf in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy()):
                raw_boxes.append([float(box[0] + x), float(box[1] + y), float(box[2] + x), float(box[3] + y)])
                raw_confs.append(float(conf))

    # Also run whole-image pass to catch larger canopies
    results_full = yolo_model.predict(source=img_bgr, conf=conf_thresh, verbose=False)[0]
    for box, conf in zip(results_full.boxes.xyxy.cpu().numpy(), results_full.boxes.conf.cpu().numpy()):
        raw_boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
        raw_confs.append(float(conf))

    # Apply Non-Maximum Suppression (NMS) to merge overlapping detections
    yolo_candidates = []
    if raw_boxes:
        merged_indices = cv2.dnn.NMSBoxes(raw_boxes, raw_confs, score_threshold=conf_thresh, nms_threshold=0.35)
        for idx in merged_indices:
            idx_num = idx[0] if isinstance(idx, (list, np.ndarray)) else idx
            b = raw_boxes[idx_num]
            yolo_candidates.append([b[0], b[1], b[2], b[3], raw_confs[idx_num]])

    # Compute ground resolution (meters per pixel)
    center_lat = (geo_meta["nw_bounds"][0] + geo_meta["se_bounds"][0]) / 2.0
    meters_per_pixel = 156543.03392 * math.cos(math.radians(center_lat)) / (2 ** zoom)

    # Run Calibrated Ground-Truth Canopy Extractor
    trees_raw, health_scores = analytics_service.detect_citrus_canopies_high_precision(
        img_bgr=img_bgr,
        poly_pixels=poly_pixels,
        yolo_boxes=yolo_candidates,
        sensitivity=float(req.sensitivity) if req.sensitivity is not None else 0.5,
        meters_per_pixel=meters_per_pixel
    )

    detected_trees = []
    tree_pixel_locs = []

    # Draw plot polygon outline on visualization canvas
    vis_img = img_bgr.copy()
    pts_int = np.array(poly_pixels, dtype=np.int32)
    cv2.polylines(vis_img, [pts_int], isClosed=True, color=(255, 200, 0), thickness=2)

    for i, t in enumerate(trees_raw):
        cx = t["pixel"]["cx"]
        cy = t["pixel"]["cy"]
        gps_lon, gps_lat = geo_meta["pixel_to_geo"](cx, cy)
        
        tree_entry = {
            "id": i + 1,
            "gps": {"lat": round(gps_lat, 6), "lon": round(gps_lon, 6)},
            "pixel": {"cx": round(float(cx), 1), "cy": round(float(cy), 1)},
            "confidence": t["confidence"],
            "canopy_diameter_m": t["canopy_diameter_m"],
            "health": t["health"]
        }
        detected_trees.append(tree_entry)
        tree_pixel_locs.append({"x": cx, "y": cy})

        # Draw green circle overlay & red centroid
        cv2.circle(vis_img, (int(cx), int(cy)), max(4, int(w / 160)), (0, 230, 70), 2)
        cv2.circle(vis_img, (int(cx), int(cy)), 2, (0, 0, 255), -1)

    # 4. Detect missing tree gaps with bare soil reflectance verification
    gaps = analytics_service.detect_planting_gaps(tree_pixel_locs, poly_pixels, img_bgr=img_bgr) or []
    for g in gaps:
        gx, gy = g["x"], g["y"]
        glon, glat = geo_meta["pixel_to_geo"](gx, gy)
        g["gps"] = {"lat": round(glat, 6), "lon": round(glon, 6)}
        # Draw subtle gap marker (Red crosshair ring)
        cv2.circle(vis_img, (int(gx), int(gy)), 5, (0, 0, 255), 1)
        cv2.drawMarker(vis_img, (int(gx), int(gy)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 8, 1)

    # 5. Economic & Yield modeling
    avg_health = float(np.mean(health_scores)) if health_scores else 75.0
    if req.market_price_per_kg:
        analytics_service.market_price = req.market_price_per_kg

    economics = analytics_service.calculate_yield_and_economics(
        tree_count=len(detected_trees),
        farm_area_ha=farm_area_ha,
        avg_health_score=avg_health
    )

    # Encode result image to base64 for direct browser rendering
    _, buffer = cv2.imencode('.jpg', vis_img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    base64_image = base64.b64encode(buffer).decode('utf-8')

    return {
        "status": "success",
        "farm_name": req.farm_name,
        "summary": {
            "total_orange_trees": len(detected_trees),
            "farm_area_ha": economics["farm_area_ha"],
            "plantation_density_trees_per_ha": economics["trees_per_ha"],
            "detected_gaps_count": len(gaps),
            "average_health_score": round(avg_health, 1),
            "estimated_annual_yield_tons": economics["total_yield_tons"],
            "estimated_production_value_inr": economics["total_production_value"]
        },
        "economics": economics,
        "trees": detected_trees,
        "gaps": gaps,
        "image_overlay_base64": f"data:image/jpeg;base64,{base64_image}"
    }

# Serve frontend static assets
os.makedirs("frontend", exist_ok=True)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
