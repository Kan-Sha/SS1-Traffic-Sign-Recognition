"""
Data Augmentation Module
========================
Module tăng cường dữ liệu cho ảnh biển báo giao thông.
Sử dụng torchvision.transforms để tạo các biến thể của ảnh training.

Sử dụng:
    from src.data.augmentation import get_train_transforms, get_test_transforms
    train_tf = get_train_transforms(image_size=64)
"""

import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch


def get_train_transforms(image_size=64, level="medium"):
    """
    Tạo augmentation transforms cho training.

    Args:
        image_size: Kích thước ảnh
        level: Mức độ augmentation ("light", "medium", "heavy")

    Returns:
        torchvision.transforms.Compose
    """
    if level == "light":
        return transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3403, 0.3121, 0.3214],
                std=[0.2724, 0.2608, 0.2669],
            ),
        ])

    elif level == "medium":
        return transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5,
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3403, 0.3121, 0.3214],
                std=[0.2724, 0.2608, 0.2669],
            ),
        ])

    elif level == "heavy":
        return transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.15, 0.15),
                scale=(0.85, 1.15),
                shear=10,
            ),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            transforms.Normalize(
                mean=[0.3403, 0.3121, 0.3214],
                std=[0.2724, 0.2608, 0.2669],
            ),
        ])

    else:
        raise ValueError(f"Invalid augmentation level: {level}. Choose 'light', 'medium', or 'heavy'.")


def get_test_transforms(image_size=64):
    """
    Tạo transforms cho validation/test (chỉ normalize, không augment).

    Args:
        image_size: Kích thước ảnh

    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.3403, 0.3121, 0.3214],
            std=[0.2724, 0.2608, 0.2669],
        ),
    ])


def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Thêm nhiễu Gaussian vào ảnh.

    Args:
        image: numpy array ảnh (H, W, C)
        mean: giá trị trung bình nhiễu
        sigma: độ lệch chuẩn nhiễu

    Returns:
        Ảnh có nhiễu
    """
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


def add_motion_blur(image, kernel_size=5):
    """
    Thêm motion blur (mô phỏng ảnh bị nhòe khi di chuyển).

    Args:
        image: numpy array ảnh
        kernel_size: kích thước kernel

    Returns:
        Ảnh bị blur
    """
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(image, -1, kernel)


def adjust_brightness(image, factor):
    """
    Điều chỉnh độ sáng ảnh.

    Args:
        image: numpy array ảnh
        factor: hệ số sáng (< 1: tối hơn, > 1: sáng hơn)

    Returns:
        Ảnh đã điều chỉnh
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def simulate_weather(image, condition="rain"):
    """
    Mô phỏng điều kiện thời tiết trên ảnh.

    Args:
        image: numpy array ảnh
        condition: "rain", "fog", "dark"

    Returns:
        Ảnh đã thêm hiệu ứng thời tiết
    """
    result = image.copy()

    if condition == "rain":
        # Thêm nhiễu dạng mưa
        rain = np.random.randint(0, 50, image.shape[:2], dtype=np.uint8)
        rain_streaks = cv2.dilate(rain, np.ones((7, 1), dtype=np.uint8))
        rain_layer = np.stack([rain_streaks] * 3, axis=-1)
        result = cv2.addWeighted(result, 0.85, rain_layer, 0.15, 0)

    elif condition == "fog":
        # Thêm hiệu ứng sương mù
        fog = np.ones_like(image, dtype=np.uint8) * 200
        result = cv2.addWeighted(result, 0.6, fog, 0.4, 0)

    elif condition == "dark":
        # Mô phỏng điều kiện ban đêm
        result = adjust_brightness(result, 0.4)
        noise = np.random.normal(0, 15, result.shape).astype(np.uint8)
        result = cv2.add(result, noise)

    return result


class OfflineAugmentor:
    """
    Augmentor offline - tạo thêm ảnh mới từ ảnh gốc để cân bằng dataset.
    Hữu ích khi một số class có ít ảnh hơn các class khác.

    Args:
        target_count: Số ảnh mục tiêu cho mỗi class
    """

    def __init__(self, target_count=2000):
        self.target_count = target_count

    def balance_dataset(self, images, labels):
        """
        Cân bằng dataset bằng cách tăng cường ảnh cho các class ít ảnh.

        Args:
            images: numpy array ảnh (N, H, W, C)
            labels: numpy array nhãn (N,)

        Returns:
            images_balanced, labels_balanced
        """
        unique_labels = np.unique(labels)
        augmented_images = list(images)
        augmented_labels = list(labels)

        for label in unique_labels:
            class_indices = np.where(labels == label)[0]
            class_count = len(class_indices)

            if class_count >= self.target_count:
                continue

            # Số ảnh cần thêm
            n_augment = self.target_count - class_count

            for _ in range(n_augment):
                # Chọn ngẫu nhiên 1 ảnh từ class
                idx = np.random.choice(class_indices)
                img = images[idx].copy()

                # Áp dụng ngẫu nhiên các augmentation
                img = self._random_augment(img)

                augmented_images.append(img)
                augmented_labels.append(label)

        return np.array(augmented_images), np.array(augmented_labels)

    def _random_augment(self, image):
        """Áp dụng các augmentation ngẫu nhiên."""
        augmentations = [
            lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            if np.random.random() > 0.7
            else img,
            lambda img: add_gaussian_noise(img, sigma=np.random.randint(5, 25)),
            lambda img: adjust_brightness(img, np.random.uniform(0.7, 1.3)),
            lambda img: add_motion_blur(img, kernel_size=np.random.choice([3, 5])),
        ]

        # Áp dụng 1-3 augmentation ngẫu nhiên
        n_augs = np.random.randint(1, 4)
        selected = np.random.choice(len(augmentations), n_augs, replace=False)

        for idx in selected:
            image = augmentations[idx](image)

        return image


if __name__ == "__main__":
    # Demo augmentation
    print("🔧 Data Augmentation Module")
    print("=" * 40)

    # Tạo ảnh giả để test
    dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    # Test augmentation transforms
    train_tf = get_train_transforms(64, "medium")
    test_tf = get_test_transforms(64)

    pil_img = Image.fromarray(dummy_image)
    augmented = train_tf(pil_img)
    print(f"✅ Augmented tensor shape: {augmented.shape}")

    # Test offline augmentor
    dummy_images = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
    dummy_labels = np.array([0] * 3 + [1] * 7)

    augmentor = OfflineAugmentor(target_count=10)
    balanced_images, balanced_labels = augmentor.balance_dataset(dummy_images, dummy_labels)
    print(f"✅ Balanced dataset: {len(balanced_images)} ảnh")
