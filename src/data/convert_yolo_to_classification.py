"""
Convert YOLO Detection Dataset → Classification Folders
=========================================================
Đọc dataset YOLO (images + label .txt files) → crop từng biển báo
→ lưu thành classification folders cho ImageFolder.

Input:  data/raw/vnts/  (images + labels dạng YOLO)
Output: data/vn_processed/train/{class_id}/
        data/vn_processed/val/{class_id}/
        data/vn_processed/test/{class_id}/

Chạy:
    python -m src.data.convert_yolo_to_classification
"""

import os
import sys
import shutil
import random
import zipfile
import csv
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
from tqdm import tqdm


def get_project_root():
    """Lấy đường dẫn gốc của project."""
    return Path(__file__).resolve().parent.parent.parent


def find_vnts_zip(data_dir):
    """
    Tìm file zip VNTS trong data/raw/.
    Tìm file zip lớn nhất có keyword 'vietnam' hoặc 'traffic' hoặc 'vnts'.
    Nếu không có, tìm file zip > 100MB.
    """
    data_dir = Path(data_dir)
    candidates = []

    for f in data_dir.glob("*.zip"):
        name_lower = f.name.lower()
        size_mb = f.stat().st_size / (1024 * 1024)

        # Ưu tiên tên file liên quan
        if any(kw in name_lower for kw in ["vietnam", "traffic", "vnts", "vietnamese"]):
            candidates.append((f, 100 + size_mb))  # high priority
        elif size_mb > 100:
            candidates.append((f, size_mb))

    if not candidates:
        return None

    # Sắp xếp theo priority
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def extract_vnts(zip_path, extract_to):
    """
    Giải nén dataset VNTS.

    Args:
        zip_path: Đường dẫn file ZIP
        extract_to: Thư mục giải nén
    """
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    print(f"  📦 Đang giải nén: {zip_path.name} ({zip_path.stat().st_size / 1e6:.0f} MB)")
    print(f"  📁 Giải nén vào: {extract_to}")

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        members = zf.namelist()
        print(f"  📊 Tổng {len(members)} files trong archive")

        for member in tqdm(members, desc="Extracting", unit="file"):
            zf.extract(member, str(extract_to))

    print(f"  ✅ Giải nén xong!")
    return extract_to


def find_dataset_dirs(extracted_dir):
    """
    Tìm thư mục chứa images và labels trong extracted data.
    Tự động detect cấu trúc: có thể là flat, train/valid/test, hay nested.

    Returns:
        List of (images_dir, labels_dir) tuples
    """
    extracted_dir = Path(extracted_dir)
    pairs = []

    # Pattern 1: train/images + train/labels (YOLO standard)
    for split_dir in extracted_dir.rglob("*"):
        if split_dir.is_dir() and split_dir.name == "images":
            labels_dir = split_dir.parent / "labels"
            if labels_dir.exists():
                pairs.append((split_dir, labels_dir))

    if pairs:
        return pairs

    # Pattern 2: Tìm bất kỳ thư mục nào chứa cả .jpg và .txt
    for d in extracted_dir.rglob("*"):
        if d.is_dir():
            jpgs = list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))
            txts = list(d.glob("*.txt"))
            if len(jpgs) > 10 and len(txts) > 10:
                pairs.append((d, d))  # images and labels in same directory

    return pairs


def parse_yolo_label(label_path, img_width, img_height):
    """
    Parse file label YOLO format.

    Format mỗi dòng: class_id x_center y_center width height
    Tất cả tọa độ normalized [0, 1].

    Returns:
        List of (class_id, x, y, w, h) in pixel coordinates
    """
    bboxes = []
    label_path = Path(label_path)

    if not label_path.exists():
        return bboxes

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

                # Convert center format to corner format
                x = int(max(0, x_center - w / 2))
                y = int(max(0, y_center - h / 2))
                w = int(min(img_width - x, w))
                h = int(min(img_height - y, h))

                if w > 5 and h > 5:  # Bỏ qua bbox quá nhỏ
                    bboxes.append((class_id, x, y, w, h))
            except (ValueError, IndexError):
                continue

    return bboxes


def find_class_names_file(extracted_dir):
    """
    Tìm file chứa tên classes (data.yaml, classes.txt, obj.names, ...).

    Returns:
        Dict mapping class_id → class_name, hoặc None
    """
    extracted_dir = Path(extracted_dir)
    class_names = {}

    # Tìm data.yaml (YOLO format)
    for yaml_file in extracted_dir.rglob("data.yaml"):
        try:
            import yaml
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if "names" in data:
                names = data["names"]
                if isinstance(names, list):
                    class_names = {i: n for i, n in enumerate(names)}
                elif isinstance(names, dict):
                    class_names = {int(k): v for k, v in names.items()}
                print(f"  ✅ Tìm thấy {len(class_names)} classes từ {yaml_file.name}")
                return class_names
        except Exception:
            continue

    # Tìm classes.txt hoặc obj.names
    for name_file in list(extracted_dir.rglob("classes.txt")) + list(extracted_dir.rglob("obj.names")):
        try:
            with open(name_file, "r", encoding="utf-8") as f:
                names = [line.strip() for line in f if line.strip()]
            if names:
                class_names = {i: n for i, n in enumerate(names)}
                print(f"  ✅ Tìm thấy {len(class_names)} classes từ {name_file.name}")
                return class_names
        except Exception:
            continue

    # Tìm _classes.csv hoặc tương tự
    for csv_file in extracted_dir.rglob("*.csv"):
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        try:
                            cid = int(row[0])
                            class_names[cid] = row[1].strip()
                        except ValueError:
                            continue
            if class_names:
                print(f"  ✅ Tìm thấy {len(class_names)} classes từ {csv_file.name}")
                return class_names
        except Exception:
            continue

    return None


def crop_and_organize(dataset_pairs, output_dir, class_names=None,
                      train_ratio=0.7, val_ratio=0.15, min_size=16, padding=5):
    """
    Crop ảnh theo bbox và tổ chức thành classification folders.

    Args:
        dataset_pairs: List of (images_dir, labels_dir)
        output_dir: Thư mục output
        class_names: Dict class_id → name (optional)
        train_ratio: Tỷ lệ train
        val_ratio: Tỷ lệ validation
        min_size: Kích thước tối thiểu crop (pixels)
        padding: Padding thêm xung quanh bbox (pixels)
    """
    output_dir = Path(output_dir)

    # Thu thập tất cả crops
    all_crops = defaultdict(list)  # class_id → list of (img_path, bbox)
    total_images = 0
    total_labels = 0
    total_bboxes = 0
    skipped = 0

    for images_dir, labels_dir in dataset_pairs:
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)

        # Tìm tất cả ảnh
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))

        print(f"\n  📂 {images_dir}")
        print(f"  📷 Tìm thấy {len(image_files)} ảnh")

        for img_path in tqdm(image_files, desc="Processing", unit="img"):
            total_images += 1

            # Tìm label file tương ứng
            label_path = labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                # Thử tìm trong cùng thư mục với ảnh
                label_path = img_path.parent / (img_path.stem + ".txt")

            if not label_path.exists():
                skipped += 1
                continue

            total_labels += 1

            # Đọc ảnh
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            h, w = img.shape[:2]

            # Parse labels
            bboxes = parse_yolo_label(label_path, w, h)

            for class_id, x, y, bw, bh in bboxes:
                # Add padding
                x = max(0, x - padding)
                y = max(0, y - padding)
                bw = min(w - x, bw + 2 * padding)
                bh = min(h - y, bh + 2 * padding)

                if bw >= min_size and bh >= min_size:
                    crop = img[y:y+bh, x:x+bw]
                    if crop.size > 0:
                        all_crops[class_id].append(crop)
                        total_bboxes += 1

    print(f"\n{'='*60}")
    print(f"📊 THỐNG KÊ CROP")
    print(f"{'='*60}")
    print(f"  📷 Tổng ảnh quét: {total_images}")
    print(f"  📋 Ảnh có label: {total_labels}")
    print(f"  ✂️  Tổng crops: {total_bboxes}")
    print(f"  ⚠️  Bỏ qua: {skipped}")
    print(f"  📂 Số classes: {len(all_crops)}")

    # Thống kê per-class
    print(f"\n  {'Class ID':<10} {'Tên':<35} {'Số ảnh':>8}")
    print(f"  {'─'*55}")
    for class_id in sorted(all_crops.keys()):
        name = class_names.get(class_id, f"Class_{class_id}") if class_names else f"Class_{class_id}"
        count = len(all_crops[class_id])
        print(f"  {class_id:<10} {name:<35} {count:>8}")

    # Split và lưu
    print(f"\n{'='*60}")
    print(f"💾 ĐANG LƯU CROPS")
    print(f"{'='*60}")

    split_counts = {"train": 0, "val": 0, "test": 0}

    for class_id, crops in all_crops.items():
        random.shuffle(crops)

        n = len(crops)

        # Đảm bảo mỗi split có ít nhất 1 ảnh
        if n < 3:
            # Quá ít → duplicate vào mỗi split
            splits = {
                "train": crops.copy(),
                "val": crops[:1],
                "test": crops[:1],
            }
        else:
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            n_test = max(1, n - n_train - n_val)
            # Adjust nếu tổng > n
            if n_train + n_val + n_test > n:
                n_train = n - n_val - n_test

            splits = {
                "train": crops[:n_train],
                "val": crops[n_train:n_train + n_val],
                "test": crops[n_train + n_val:],
            }

        for split_name, split_crops in splits.items():
            class_dir = output_dir / split_name / f"{class_id:02d}"
            class_dir.mkdir(parents=True, exist_ok=True)

            for i, crop in enumerate(split_crops):
                # Resize về 64x64 (consistent với GTSRB)
                resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
                save_path = class_dir / f"{class_id:02d}_{i:05d}.jpg"
                cv2.imwrite(str(save_path), resized)
                split_counts[split_name] += 1

    print(f"\n  ✅ Train: {split_counts['train']} ảnh")
    print(f"  ✅ Val:   {split_counts['val']} ảnh")
    print(f"  ✅ Test:  {split_counts['test']} ảnh")
    print(f"  ✅ Tổng:  {sum(split_counts.values())} ảnh")
    print(f"  📁 Output: {output_dir}")

    return all_crops, class_names


def save_class_names(class_names, output_dir):
    """Lưu mapping class names ra file JSON."""
    import json
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "vn_class_names.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"  💾 Saved class names → {path}")
    return path


def main():
    """Main: Tìm zip → giải nén → crop → organize."""
    random.seed(42)

    project_root = get_project_root()
    raw_dir = project_root / "data" / "raw"
    vnts_extracted = raw_dir / "vnts"
    output_dir = project_root / "data" / "vn_processed"

    print("=" * 60)
    print("🇻🇳 VNTS Dataset Converter")
    print("  YOLO Detection → Classification Folders")
    print("=" * 60)

    # Step 1: Tìm file zip
    if not vnts_extracted.exists() or not any(vnts_extracted.iterdir()):
        zip_path = find_vnts_zip(raw_dir)
        if zip_path is None:
            print(f"\n  ❌ Không tìm thấy file zip dataset VN trong {raw_dir}")
            print(f"  📥 Tải dataset từ:")
            print(f"     https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs")
            print(f"  📁 Đặt file zip vào: {raw_dir}")
            sys.exit(1)

        print(f"\n  📦 Tìm thấy: {zip_path.name}")
        extract_vnts(zip_path, vnts_extracted)
    else:
        print(f"\n  ✅ Dataset đã giải nén: {vnts_extracted}")

    # Step 2: Tìm structure
    print(f"\n{'='*60}")
    print(f"🔍 PHÂN TÍCH CẤU TRÚC DATASET")
    print(f"{'='*60}")

    pairs = find_dataset_dirs(vnts_extracted)
    if not pairs:
        print(f"  ❌ Không tìm thấy cặp images/labels trong {vnts_extracted}")
        print(f"  📂 Cấu trúc hiện tại:")
        for item in sorted(vnts_extracted.rglob("*"))[:30]:
            rel = item.relative_to(vnts_extracted)
            prefix = "📁" if item.is_dir() else "📄"
            print(f"     {prefix} {rel}")
        sys.exit(1)

    print(f"  ✅ Tìm thấy {len(pairs)} cặp (images, labels):")
    for img_dir, lbl_dir in pairs:
        print(f"     📷 {img_dir}")
        print(f"     📋 {lbl_dir}")

    # Step 3: Tìm class names
    print(f"\n{'='*60}")
    print(f"📋 TÌM CLASS NAMES")
    print(f"{'='*60}")

    class_names = find_class_names_file(vnts_extracted)
    if class_names is None:
        print("  ⚠️ Không tìm thấy file class names. Sẽ dùng ID số.")
        class_names = {}

    # Step 4: Crop và organize
    print(f"\n{'='*60}")
    print(f"✂️  CROP & ORGANIZE")
    print(f"{'='*60}")

    all_crops, class_names = crop_and_organize(
        dataset_pairs=pairs,
        output_dir=output_dir,
        class_names=class_names,
        train_ratio=0.7,
        val_ratio=0.15,
        min_size=16,
        padding=5,
    )

    # Step 5: Lưu class names
    if class_names:
        save_class_names(
            {str(k): v for k, v in class_names.items()},
            output_dir
        )
    else:
        # Auto-generate class names từ class IDs
        auto_names = {str(k): f"Bien_bao_VN_{k:02d}" for k in all_crops.keys()}
        save_class_names(auto_names, output_dir)

    print(f"\n{'='*60}")
    print(f"✅ HOÀN TẤT!")
    print(f"{'='*60}")
    print(f"  📁 Output: {output_dir}")
    print(f"  🔮 Bước tiếp: Chạy training")
    print(f"     python -m src.models.train --dataset vn")


if __name__ == "__main__":
    main()
