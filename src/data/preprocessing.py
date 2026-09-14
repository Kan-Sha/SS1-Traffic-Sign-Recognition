"""
Image Preprocessing Module
===========================
Module tiền xử lý ảnh biển báo giao thông: resize, normalize, CLAHE, chia dataset.

Sử dụng:
    from src.data.preprocessing import TrafficSignPreprocessor
    preprocessor = TrafficSignPreprocessor(image_size=64)
    preprocessor.prepare_dataset("data/raw", "data/processed")
"""

import os
import csv
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import yaml


class TrafficSignDataset(Dataset):
    """
    PyTorch Dataset cho biển báo giao thông.

    Args:
        images: numpy array ảnh (N, H, W, C)
        labels: numpy array nhãn (N,)
        transform: torchvision transforms
    """

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Chuyển numpy array sang PIL Image
        image = Image.fromarray(image.astype(np.uint8))

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label


class TrafficSignPreprocessor:
    """
    Tiền xử lý ảnh biển báo giao thông.

    Args:
        image_size: Kích thước ảnh đầu ra (mặc định 64x64)
        apply_clahe: Có áp dụng CLAHE hay không
    """

    def __init__(self, image_size=64, apply_clahe=True):
        self.image_size = image_size
        self.apply_clahe = apply_clahe
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def resize_image(self, image, size=None):
        """
        Resize ảnh về kích thước chuẩn.

        Args:
            image: numpy array ảnh
            size: kích thước đích (mặc định self.image_size)

        Returns:
            Ảnh đã resize
        """
        if size is None:
            size = self.image_size
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    def apply_clahe_enhancement(self, image):
        """
        Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Tăng cường contrast cho ảnh, đặc biệt hữu ích cho ảnh tối hoặc thiếu sáng.

        Args:
            image: numpy array ảnh BGR

        Returns:
            Ảnh đã tăng cường contrast
        """
        # Chuyển sang LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Áp dụng CLAHE lên kênh L (Lightness)
        l_channel = self.clahe.apply(l_channel)

        # Ghép lại các kênh
        lab = cv2.merge([l_channel, a_channel, b_channel])

        # Chuyển về BGR
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def normalize_image(self, image):
        """
        Normalize pixel values về [0, 1].

        Args:
            image: numpy array ảnh

        Returns:
            Ảnh đã normalize
        """
        return image.astype(np.float32) / 255.0

    def preprocess_single(self, image):
        """
        Pipeline tiền xử lý cho 1 ảnh.

        Args:
            image: numpy array ảnh BGR

        Returns:
            Ảnh đã tiền xử lý (RGB, resized)
        """
        # Resize
        image = self.resize_image(image)

        # CLAHE
        if self.apply_clahe:
            image = self.apply_clahe_enhancement(image)

        # Chuyển BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def load_gtsrb_training_data(self, data_dir):
        """
        Tải dữ liệu training GTSRB từ thư mục.

        Args:
            data_dir: Đường dẫn đến thư mục GTSRB

        Returns:
            images: numpy array (N, H, W, C)
            labels: numpy array (N,)
        """
        data_dir = Path(data_dir)
        images = []
        labels = []

        # Tìm thư mục GTSRB
        gtsrb_dir = None
        for candidate in [
            data_dir / "GTSRB" / "Final_Training" / "Images",
            data_dir / "GTSRB" / "Training",
            data_dir / "Final_Training" / "Images",
            data_dir,
        ]:
            if candidate.exists():
                gtsrb_dir = candidate
                break

        if gtsrb_dir is None:
            raise FileNotFoundError(
                f"Không tìm thấy thư mục GTSRB trong {data_dir}. "
                "Vui lòng chạy: python -m src.data.download_dataset"
            )

        print(f"📂 Đang tải dữ liệu từ: {gtsrb_dir}")

        # Duyệt 43 classes (00000 - 00042)
        for class_id in tqdm(range(43), desc="Đang đọc ảnh"):
            class_dir = gtsrb_dir / f"{class_id:05d}"

            if not class_dir.exists():
                print(f"  ⚠ Không tìm thấy thư mục class {class_id}")
                continue

            # Đọc annotations CSV nếu có
            csv_files = list(class_dir.glob("*.csv"))
            if csv_files:
                with open(csv_files[0], "r") as f:
                    reader = csv.reader(f, delimiter=";")
                    next(reader)  # Skip header
                    for row in reader:
                        image_path = class_dir / row[0]
                        if image_path.exists():
                            img = cv2.imread(str(image_path))
                            if img is not None:
                                img = self.preprocess_single(img)
                                images.append(img)
                                labels.append(class_id)
            else:
                # Đọc trực tiếp từ thư mục
                for img_path in class_dir.glob("*.ppm"):
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = self.preprocess_single(img)
                        images.append(img)
                        labels.append(class_id)

                for img_path in class_dir.glob("*.png"):
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = self.preprocess_single(img)
                        images.append(img)
                        labels.append(class_id)

        images = np.array(images, dtype=np.uint8)
        labels = np.array(labels, dtype=np.int64)

        print(f"✅ Đã tải: {len(images)} ảnh, {len(set(labels))} classes")
        return images, labels

    def split_dataset(self, images, labels, train_ratio=0.7, val_ratio=0.15, seed=42):
        """
        Chia dataset thành train/validation/test.

        Args:
            images: numpy array ảnh
            labels: numpy array nhãn
            train_ratio: tỷ lệ train (mặc định 0.7)
            val_ratio: tỷ lệ validation (mặc định 0.15)
            seed: random seed

        Returns:
            Dict chứa {train, val, test} với images và labels
        """
        test_ratio = 1.0 - train_ratio - val_ratio

        # Chia train + (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels,
            test_size=(val_ratio + test_ratio),
            random_state=seed,
            stratify=labels,
        )

        # Chia val + test
        val_relative = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1.0 - val_relative),
            random_state=seed,
            stratify=y_temp,
        )

        print(f"📊 Phân chia dataset:")
        print(f"   Train: {len(X_train)} ảnh ({train_ratio*100:.0f}%)")
        print(f"   Val:   {len(X_val)} ảnh ({val_ratio*100:.0f}%)")
        print(f"   Test:  {len(X_test)} ảnh ({test_ratio*100:.0f}%)")

        return {
            "train": {"images": X_train, "labels": y_train},
            "val": {"images": X_val, "labels": y_val},
            "test": {"images": X_test, "labels": y_test},
        }

    def create_dataloaders(self, data_splits, batch_size=64, train_transform=None,
                           test_transform=None, num_workers=0):
        """
        Tạo DataLoaders cho train/val/test.

        Args:
            data_splits: Dict từ split_dataset()
            batch_size: kích thước batch
            train_transform: transforms cho training
            test_transform: transforms cho val/test
            num_workers: số worker cho DataLoader

        Returns:
            Dict chứa DataLoaders {train, val, test}
        """
        if test_transform is None:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.3403, 0.3121, 0.3214],
                    std=[0.2724, 0.2608, 0.2669],
                ),
            ])

        if train_transform is None:
            train_transform = test_transform

        dataloaders = {}
        for split_name, split_data in data_splits.items():
            transform = train_transform if split_name == "train" else test_transform
            shuffle = split_name == "train"

            dataset = TrafficSignDataset(
                images=split_data["images"],
                labels=split_data["labels"],
                transform=transform,
            )

            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
            )

        return dataloaders

    def prepare_dataset(self, raw_dir, processed_dir=None):
        """
        Pipeline hoàn chỉnh: load → preprocess → split → save.

        Args:
            raw_dir: Thư mục chứa dữ liệu gốc
            processed_dir: Thư mục lưu dữ liệu đã xử lý

        Returns:
            Dict chứa data splits
        """
        # Load
        images, labels = self.load_gtsrb_training_data(raw_dir)

        # Split
        data_splits = self.split_dataset(images, labels)

        # Save nếu cần
        if processed_dir:
            processed_dir = Path(processed_dir)
            processed_dir.mkdir(parents=True, exist_ok=True)

            for split_name, split_data in data_splits.items():
                np.save(
                    str(processed_dir / f"{split_name}_images.npy"),
                    split_data["images"],
                )
                np.save(
                    str(processed_dir / f"{split_name}_labels.npy"),
                    split_data["labels"],
                )

            print(f"\n💾 Đã lưu dữ liệu xử lý vào: {processed_dir}")

        return data_splits


def get_default_transforms(image_size=64):
    """
    Lấy transforms mặc định cho train và test.

    Args:
        image_size: kích thước ảnh

    Returns:
        Tuple (train_transform, test_transform)
    """
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.3403, 0.3121, 0.3214],
            std=[0.2724, 0.2608, 0.2669],
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.3403, 0.3121, 0.3214],
            std=[0.2724, 0.2608, 0.2669],
        ),
    ])

    return train_transform, test_transform


if __name__ == "__main__":
    preprocessor = TrafficSignPreprocessor(image_size=64)
    preprocessor.prepare_dataset("data/raw", "data/processed")
