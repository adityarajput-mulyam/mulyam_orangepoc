"""
Verification audit script for datasets.
"""
from pathlib import Path
from PIL import Image

def audit_dataset():
    root = Path("datasets/neon_trees")
    splits = ["train", "val", "test"]
    
    print("=" * 65)
    print("DATASET INTEGRITY & VALIDATION AUDIT")
    print("=" * 65)
    
    total_images = 0
    total_boxes = 0
    corrupted_images = 0
    invalid_boxes = 0
    
    for split in splits:
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        
        images = list(img_dir.glob("*.jpg"))
        labels = list(lbl_dir.glob("*.txt"))
        
        split_boxes = 0
        
        for img_p in images:
            try:
                with Image.open(img_p) as im:
                    im.verify()
            except Exception:
                corrupted_images += 1
                
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if lbl_p.exists():
                with open(lbl_p, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls, x, y, w, h = map(float, parts)
                            if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
                                split_boxes += 1
                            else:
                                invalid_boxes += 1
                                
        total_images += len(images)
        total_boxes += split_boxes
        print(f"Split [{split.upper():<5}]: {len(images):>3} images | {len(labels):>3} label files | {split_boxes:>4} valid tree boxes")
        
    print("-" * 65)
    print(f"Total Valid Images Checked     : {total_images}")
    print(f"Total Labeled Tree Crowns      : {total_boxes}")
    print(f"Corrupted Images               : {corrupted_images}")
    print(f"Invalid Bounding Boxes         : {invalid_boxes}")
    print(f"data.yaml Config Exists        : {(root / 'data.yaml').exists()}")
    print("=" * 65)

if __name__ == "__main__":
    audit_dataset()
