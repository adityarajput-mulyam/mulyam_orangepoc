"""
Downloads the NEON Tree Evaluation benchmark dataset from Weecology GitHub,
converts Pascal VOC / XML bounding box annotations to YOLO format,
and creates a train/val/test split with data.yaml.
"""
import os
import xml.etree.ElementTree as ET
import random
import shutil
import requests
from pathlib import Path
from PIL import Image

DATASET_ROOT = Path("datasets/neon_trees")
TRAIN_IMG = DATASET_ROOT / "train" / "images"
TRAIN_LBL = DATASET_ROOT / "train" / "labels"
VAL_IMG = DATASET_ROOT / "val" / "images"
VAL_LBL = DATASET_ROOT / "val" / "labels"
TEST_IMG = DATASET_ROOT / "test" / "images"
TEST_LBL = DATASET_ROOT / "test" / "labels"

RAW_DIR = DATASET_ROOT / "raw"

for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL, TEST_IMG, TEST_LBL, RAW_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GITHUB_API_BASE = "https://api.github.com/repos/weecology/NeonTreeEvaluation/contents"
RAW_BASE_RGB = "https://raw.githubusercontent.com/weecology/NeonTreeEvaluation/master/evaluation/RGB"
RAW_BASE_ANN = "https://raw.githubusercontent.com/weecology/NeonTreeEvaluation/master/annotations"

def download_neon_benchmark(max_samples=200):
    print("Fetching list of annotated samples from GitHub...")
    r = requests.get(f"{GITHUB_API_BASE}/annotations")
    if r.status_code != 200:
        print(f"Failed to fetch annotations list: {r.status_code}")
        return []
    
    ann_files = [f['name'] for f in r.json() if f['name'].endswith('.xml')]
    print(f"Found {len(ann_files)} annotated samples.")
    
    if max_samples:
        ann_files = ann_files[:max_samples]
        
    downloaded_samples = []
    session = requests.Session()
    
    for i, xml_name in enumerate(ann_files, 1):
        base_stem = Path(xml_name).stem
        tif_name = f"{base_stem}.tif"
        
        xml_path = RAW_DIR / xml_name
        tif_path = RAW_DIR / tif_name
        jpg_path = RAW_DIR / f"{base_stem}.jpg"
        
        # Download XML
        if not xml_path.exists():
            r_xml = session.get(f"{RAW_BASE_ANN}/{xml_name}")
            if r_xml.status_code == 200:
                with open(xml_path, 'w', encoding='utf-8') as f:
                    f.write(r_xml.text)
            else:
                continue
                
        # Download TIF
        if not tif_path.exists() and not jpg_path.exists():
            r_tif = session.get(f"{RAW_BASE_RGB}/{tif_name}")
            if r_tif.status_code == 200:
                with open(tif_path, 'wb') as f:
                    f.write(r_tif.content)
            else:
                continue
                
        # Convert TIF to JPG for YOLO compatibility
        if tif_path.exists() and not jpg_path.exists():
            try:
                with Image.open(tif_path) as img:
                    rgb_img = img.convert('RGB')
                    rgb_img.save(jpg_path, 'JPEG', quality=95)
                tif_path.unlink() # Save disk space
            except Exception as e:
                print(f"Error converting {tif_name}: {e}")
                continue
                
        downloaded_samples.append(base_stem)
        if i % 20 == 0 or i == len(ann_files):
            print(f"Progress: {i}/{len(ann_files)} samples processed.")
            
    return downloaded_samples

def xml_to_yolo(xml_path: Path, img_width: int, img_height: int):
    """Parses XML bounding boxes and returns YOLO lines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_lines = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        if bndbox is None:
            continue
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        # Clamp to image boundaries
        xmin = max(0, min(xmin, img_width))
        xmax = max(0, min(xmax, img_width))
        ymin = max(0, min(ymin, img_height))
        ymax = max(0, min(ymax, img_height))
        
        box_w = xmax - xmin
        box_h = ymax - ymin
        if box_w <= 0 or box_h <= 0:
            continue
            
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        w = box_w / img_width
        h = box_h / img_height
        
        # Class 0: Tree / Tree Crown
        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        
    return yolo_lines

def build_yolo_splits(samples):
    random.seed(42)
    random.shuffle(samples)
    
    n = len(samples)
    n_train = int(n * 0.75)
    n_val = int(n * 0.15)
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train+n_val]
    test_samples = samples[n_train+n_val:]
    
    splits = [
        ("train", train_samples, TRAIN_IMG, TRAIN_LBL),
        ("val", val_samples, VAL_IMG, VAL_LBL),
        ("test", test_samples, TEST_IMG, TEST_LBL),
    ]
    
    total_boxes = 0
    for split_name, sample_list, img_dest, lbl_dest in splits:
        for stem in sample_list:
            jpg_file = RAW_DIR / f"{stem}.jpg"
            xml_file = RAW_DIR / f"{stem}.xml"
            if not jpg_file.exists() or not xml_file.exists():
                continue
                
            with Image.open(jpg_file) as img:
                w, h = img.size
                
            yolo_lines = xml_to_yolo(xml_file, w, h)
            if not yolo_lines:
                continue
                
            total_boxes += len(yolo_lines)
            
            # Copy image
            shutil.copy2(jpg_file, img_dest / f"{stem}.jpg")
            
            # Write label
            lbl_file = lbl_dest / f"{stem}.txt"
            with open(lbl_file, 'w', encoding='utf-8') as f:
                f.writelines(yolo_lines)
                
        print(f"Split [{split_name}]: {len(sample_list)} images copied.")
        
    # Write data.yaml
    yaml_content = f"""# NEON Tree Crowns Dataset for YOLO
path: {DATASET_ROOT.resolve().as_posix()}
train: train/images
val: val/images
test: test/images

nc: 1
names: ['tree_crown']
"""
    yaml_path = DATASET_ROOT / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
        
    print(f"\nCreated {yaml_path}")
    print(f"Total labeled tree crowns across splits: {total_boxes}")

if __name__ == "__main__":
    print("=== Downloading and Preparing NEON Tree Crowns Benchmark Dataset ===")
    samples = download_neon_benchmark(max_samples=226)
    if samples:
        build_yolo_splits(samples)
        print("=== Dataset Preparation Complete ===")
