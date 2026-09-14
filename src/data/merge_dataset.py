"""
Merge Dataset Mới vào VN Processed
====================================
Script merge ảnh từ dataset mới (archive_1) vào data/vn_processed/
Tự động mapping class name → folder số, resize, và re-split train/val/test.

Chạy:
    python -m src.data.merge_dataset
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path
from PIL import Image
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_class_mapping(vn_processed_dir):
    """Load mapping từ class name → folder number."""
    json_path = vn_processed_dir / "vn_class_names.json"
    if not json_path.exists():
        print(f"  ❌ Không tìm thấy {json_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Đảo ngược: class_name → folder_number (2 chữ số)
    name_to_folder = {}
    for idx, name in data.items():
        name_to_folder[name] = f"{int(idx):02d}"

    return name_to_folder, data


def count_images_in_dir(directory):
    """Đếm số ảnh trong một thư mục."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}
    count = 0
    if directory.exists():
        for f in directory.iterdir():
            if f.suffix.lower() in exts:
                count += 1
    return count


def merge_new_images(new_dataset_dir, vn_processed_dir, image_size=64):
    """
    Merge ảnh mới vào các thư mục train/val/test của vn_processed.

    Args:
        new_dataset_dir: Thư mục chứa ảnh mới (tổ chức theo class name)
        vn_processed_dir: Thư mục vn_processed hiện tại
        image_size: Kích thước resize ảnh
    """
    name_to_folder, idx_to_name = load_class_mapping(vn_processed_dir)

    print("=" * 60)
    print("🔄 MERGE DATASET MỚI VÀO VN_PROCESSED")
    print("=" * 60)
    print(f"  📁 Nguồn: {new_dataset_dir}")
    print(f"  📁 Đích:  {vn_processed_dir}")
    print()

    # Thống kê trước khi merge
    print("📊 THỐNG KÊ TRƯỚC MERGE:")
    stats_before = {}
    for split in ["train", "val", "test"]:
        split_dir = vn_processed_dir / split
        total = 0
        for folder in sorted(split_dir.iterdir()):
            if folder.is_dir():
                total += count_images_in_dir(folder)
        stats_before[split] = total
        print(f"  {split}: {total} ảnh")

    # Duyệt từng class trong dataset mới
    total_merged = 0
    total_skipped = 0
    class_stats = {}

    new_classes = sorted([d for d in new_dataset_dir.iterdir() if d.is_dir()])

    for class_dir in new_classes:
        class_name = class_dir.name

        if class_name not in name_to_folder:
            print(f"  ⚠️ SKIP: {class_name} (không có trong mapping)")
            total_skipped += count_images_in_dir(class_dir)
            continue

        folder_num = name_to_folder[class_name]

        # Lấy tất cả ảnh trong class mới
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}
        new_images = [f for f in class_dir.iterdir() if f.suffix.lower() in exts]

        if not new_images:
            continue

        # Shuffle và split 70/15/15
        random.seed(42)
        random.shuffle(new_images)

        n = len(new_images)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)

        splits = {
            "train": new_images[:n_train],
            "val": new_images[n_train:n_train + n_val],
            "test": new_images[n_train + n_val:],
        }

        class_merged = 0
        for split_name, images in splits.items():
            dest_dir = vn_processed_dir / split_name / folder_num
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Tìm số file hiện có để tránh trùng tên
            existing = count_images_in_dir(dest_dir)

            for i, img_path in enumerate(images):
                try:
                    # Load và resize ảnh
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize((image_size, image_size), Image.LANCZOS)

                    # Lưu với tên unique
                    new_name = f"merged_{existing + i:05d}.jpg"
                    save_path = dest_dir / new_name
                    img.save(save_path, "JPEG", quality=95)
                    class_merged += 1
                except Exception as e:
                    print(f"  ❌ Lỗi xử lý {img_path.name}: {e}")

        total_merged += class_merged
        class_stats[class_name] = class_merged
        print(f"  ✅ [{folder_num}] {class_name}: +{class_merged} ảnh")

    # Thống kê sau merge
    print()
    print("=" * 60)
    print("📊 KẾT QUẢ MERGE:")
    print("=" * 60)
    print(f"  ✅ Tổng ảnh merge: {total_merged}")
    print(f"  ⚠️ Ảnh skip:      {total_skipped}")
    print(f"  📋 Classes merge:  {len(class_stats)}/39")
    print()

    print("📊 THỐNG KÊ SAU MERGE:")
    for split in ["train", "val", "test"]:
        split_dir = vn_processed_dir / split
        total = 0
        for folder in sorted(split_dir.iterdir()):
            if folder.is_dir():
                total += count_images_in_dir(folder)
        print(f"  {split}: {stats_before[split]} → {total} ảnh (+{total - stats_before[split]})")

    # In chi tiết các class ít ảnh nhất
    print()
    print("📊 CÁC CLASS ÍT ẢNH NHẤT (sau merge):")
    class_counts = []
    train_dir = vn_processed_dir / "train"
    for folder in sorted(train_dir.iterdir()):
        if folder.is_dir():
            idx = folder.name
            name = idx_to_name.get(str(int(idx)), idx)
            count = count_images_in_dir(folder)
            class_counts.append((idx, name, count))

    class_counts.sort(key=lambda x: x[2])
    for idx, name, count in class_counts[:10]:
        marker = "🔴" if count < 30 else "🟡" if count < 100 else "🟢"
        print(f"  {marker} [{idx}] {name}: {count} ảnh")

    print()
    print("🎉 MERGE HOÀN TẤT!")
    print("  Bước tiếp theo: python -m src.models.train_vn --epochs 30")


def main():
    # Đường dẫn dataset mới
    new_dataset = PROJECT_ROOT / "data" / "raw" / "archive_1" / "dataset_cropped_for_lenet" / "dataset_cropped_for_lenet"
    vn_processed = PROJECT_ROOT / "data" / "vn_processed"

    if not new_dataset.exists():
        print(f"❌ Không tìm thấy dataset mới tại: {new_dataset}")
        print("  Giải nén archive (1).zip vào data/raw/archive_1/ trước.")
        sys.exit(1)

    if not vn_processed.exists():
        print(f"❌ Không tìm thấy vn_processed tại: {vn_processed}")
        sys.exit(1)

    merge_new_images(new_dataset, vn_processed)


if __name__ == "__main__":
    main()
