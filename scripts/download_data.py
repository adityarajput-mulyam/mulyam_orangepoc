"""
Unified dataset downloader for Citrus and Tree Crown Detection.
Downloads and prepares benchmark datasets (NEON tree crowns & Roboflow if API key provided).
"""
import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.prepare_neon_dataset import download_neon_benchmark, build_yolo_splits

def setup_dataset(roboflow_api_key: str = None):
    print("=" * 60)
    print("Starting Tree Crown & Citrus Dataset Setup")
    print("=" * 60)
    
    # 1. Check if NEON dataset already prepared
    neon_yaml = Path("datasets/neon_trees/data.yaml")
    if neon_yaml.exists():
        print(f"[FOUND] Benchmark dataset already prepared at: {neon_yaml.resolve()}")
    else:
        print("[INFO] Preparing NEON Tree benchmark dataset...")
        samples = download_neon_benchmark(max_samples=226)
        if samples:
            build_yolo_splits(samples)
            print(f"[SUCCESS] Prepared NEON dataset at: {neon_yaml.resolve()}")
            
    # 2. Optional: Roboflow Citrus dataset if API key is provided
    if roboflow_api_key or os.environ.get("ROBOFLOW_API_KEY"):
        key = roboflow_api_key or os.environ.get("ROBOFLOW_API_KEY")
        print(f"\n[INFO] Roboflow API key provided. Fetching citrus-specific dataset...")
        try:
            import importlib
            roboflow_module = importlib.import_module("roboflow")
            Roboflow = getattr(roboflow_module, "Roboflow")
            rf = Roboflow(api_key=key)
            project = rf.workspace("aviral").project("tree-hoqde")
            version = project.version(1)
            dataset = version.download("yolov8", location="datasets/citrus_roboflow")
            print(f"[SUCCESS] Downloaded Roboflow dataset to: {dataset.location}")
        except Exception as e:
            print(f"[WARNING] Could not fetch from Roboflow: {e}")
            
    print("=" * 60)
    print("Dataset setup complete and ready for training!")
    print("=" * 60)

if __name__ == "__main__":
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    setup_dataset(roboflow_api_key=api_key)
