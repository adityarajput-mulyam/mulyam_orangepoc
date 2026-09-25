# Citrus Tree Counting & Farm Analytics PoC

Satellite-based citrus tree detection, canopy counting, tree health estimation, and farm economics modeling using YOLOv8 and FastAPI.

---

## 🎯 Model Accuracy & Benchmarks

Trained on aerial/satellite tree crown imagery using **YOLOv8n** (`imgsz=640`, 74 epochs with early stopping):

| Metric | Score |
| :--- | :--- |
| **mAP@50** | **59.4%** |
| **mAP@50-95** | **22.5%** |
| **Precision** | **66.6%** |
| **Recall** | **56.1%** |

*Weights location:* `runs/detect/runs/tree_crown_detection/yolov8_citrus_poc-2/weights/best.pt`

---

## 🚀 Quick Setup

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Clone & Virtual Environment
```bash
git clone git@github.com-work:adityarajput-mulyam/mulyam_orangepoc.git
cd mulyam_orangepoc

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install fastapi uvicorn ultralytics opencv-python numpy shapely requests pillow
```

### 4. Run Application
```bash
python app.py
```
Or with uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Access the interactive web UI at: **`http://localhost:8000`**

---

## 🏗️ Project Structure

- `app.py`: FastAPI backend, tile fetching, inference pipeline, economic calculations.
- `frontend/`: Map interface (polygon drawing, live detection overlay, farm dashboard).
- `services/`:
  - `satellite_service.py`: Google/Mapbox slippy tile fetching and stitching.
  - `analytics.py`: Yield, canopy health, density, and financial modeling.
- `train_yolo.py`: Model training script.
- `scripts/`: Dataset prep and CLI inference helpers.
