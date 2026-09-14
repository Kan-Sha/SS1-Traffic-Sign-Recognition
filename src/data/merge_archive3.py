"""
Merge Archive 3 (YOLO format) vào VN Processed
=================================================
Crop bbox từ archive_3 → merge vào data/vn_processed/

Chạy:
    python -m src.data.merge_archive3
"""

import os
import sys
import json
import random
from pathlib import Path
from collections import defaultdict

import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_yolo_label(label_path, img_width, img_height, padding=5):
    """Parse YOLO label → list of (class_id, crop_region)."""
    bboxes = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                w = float(parts[3]) * img_width
                h = float(parts[4]) * img_height

                x = int(max(0, x_center - w / 2 - padding))
                y = int(max(0, y_center - h / 2 - padding))
                w = int(min(img_width - x, w + 2 * padding))
                h = int(min(img_height - y, h + 2 * padding))

                if w > 10 and h > 10:
                    bboxes.append((class_id, x, y, w, h))
            except (ValueError, IndexError):
                continue
    return bboxes


def main():
    random.seed(42)

    images_dir = PROJECT_ROOT / "data" / "raw" / "archive_3" / "archive" / "images"
    labels_dir = PROJECT_ROOT / "data" / "raw" / "archive_3" / "archive" / "labels"
    classes_file = PROJECT_ROOT / "data" / "raw" / "archive_3" / "archive" / "classes.txt"
    vn_processed = PROJECT_ROOT / "data" / "vn_processed"

    # Load class names tu archive_3
    with open(classes_file, "r", encoding="utf-8") as f:
        new_class_names = {i: line.strip() for i, line in enumerate(f) if line.strip()}
    print(f"Archive 3: {len(new_class_names)} classes")

    # Load mapping hien tai
    with open(vn_processed / "vn_class_names.json", "r", encoding="utf-8") as f:
        current_mapping = json.load(f)

    # name -> folder number
    name_to_folder = {}
    for idx, name in current_mapping.items():
        name_to_folder[name] = f"{int(idx):02d}"

    print(f"Current VN dataset: {len(current_mapping)} classes")

    # Crop images from archive_3
    print("\n" + "=" * 60)
    print("Cropping images from archive_3...")
    print("=" * 60)

    crops_by_class = defaultdict(list)  # class_name -> list of crops
    total_crops = 0
    skipped = 0

    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    print(f"  Found {len(image_files)} images")

    for img_path in tqdm(image_files, desc="Cropping"):
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        h, w = img.shape[:2]
        bboxes = parse_yolo_label(label_path, w, h)

        for class_id, x, y, bw, bh in bboxes:
            if class_id not in new_class_names:
                continue

            class_name = new_class_names[class_id]

            # Chi merge class trung voi dataset hien tai
            if class_name not in name_to_folder:
                continue

            crop = img[y:y+bh, x:x+bw]
            if crop.size > 0:
                resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
                crops_by_class[class_name].append(resized)
                total_crops += 1

    print(f"\n  Total crops: {total_crops}")
    print(f"  Skipped: {skipped}")
    print(f"  Matching classes: {len(crops_by_class)}")

    # Count before
    print("\nBEFORE MERGE:")
    for split in ["train", "val", "test"]:
        split_dir = vn_processed / split
        total = sum(len(os.listdir(d)) for d in split_dir.iterdir() if d.is_dir())
        print(f"  {split}: {total}")

    # Merge crops vao vn_processed
    print("\nMerging...")
    total_merged = 0

    for class_name, crops in sorted(crops_by_class.items()):
        folder_num = name_to_folder[class_name]

        random.shuffle(crops)
        n = len(crops)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)

        splits = {
            "train": crops[:n_train],
            "val": crops[n_train:n_train + n_val],
            "test": crops[n_train + n_val:],
        }

        class_merged = 0
        for split_name, split_crops in splits.items():
            dest_dir = vn_processed / split_name / folder_num
            dest_dir.mkdir(parents=True, exist_ok=True)

            existing = len(os.listdir(dest_dir))

            for i, crop_img in enumerate(split_crops):
                new_name = f"arch3_{existing + i:05d}.jpg"
                save_path = dest_dir / new_name
                cv2.imwrite(str(save_path), crop_img)
                class_merged += 1

        total_merged += class_merged
        print(f"  [{folder_num}] {class_name}: +{class_merged}")

    # Count after
    print(f"\nTotal merged: {total_merged}")
    print("\nAFTER MERGE:")
    for split in ["train", "val", "test"]:
        split_dir = vn_processed / split
        total = sum(len(os.listdir(d)) for d in split_dir.iterdir() if d.is_dir())
        print(f"  {split}: {total}")

    print("\nDone! Ready to train: python -m src.models.train_vn --epochs 30")


if __name__ == "__main__":
    main()
