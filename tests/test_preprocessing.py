"""
Test Preprocessing Module
=========================
Unit tests cho module tiền xử lý ảnh.
"""

import os
import sys
import numpy as np
import pytest
from pathlib import Path

# Thêm project root vào path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import TrafficSignPreprocessor, TrafficSignDataset


class TestTrafficSignPreprocessor:
    """Test class cho TrafficSignPreprocessor."""

    def setup_method(self):
        """Setup cho mỗi test."""
        self.preprocessor = TrafficSignPreprocessor(image_size=64)

    def test_resize_image(self):
        """Test resize ảnh."""
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        resized = self.preprocessor.resize_image(image, 64)
        assert resized.shape == (64, 64, 3), f"Expected (64, 64, 3), got {resized.shape}"

    def test_resize_non_square(self):
        """Test resize ảnh không vuông."""
        image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        resized = self.preprocessor.resize_image(image, 32)
        assert resized.shape == (32, 32, 3)

    def test_clahe_enhancement(self):
        """Test CLAHE enhancement."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        enhanced = self.preprocessor.apply_clahe_enhancement(image)
        assert enhanced.shape == image.shape
        assert enhanced.dtype == np.uint8

    def test_normalize_image(self):
        """Test normalize ảnh."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        normalized = self.preprocessor.normalize_image(image)
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert normalized.dtype == np.float32

    def test_preprocess_single(self):
        """Test pipeline tiền xử lý 1 ảnh."""
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        processed = self.preprocessor.preprocess_single(image)
        assert processed.shape == (64, 64, 3)

    def test_split_dataset(self):
        """Test chia dataset."""
        n_samples = 100
        images = np.random.randint(0, 255, (n_samples, 64, 64, 3), dtype=np.uint8)
        labels = np.repeat(np.arange(10), 10)  # 10 classes, 10 samples each

        splits = self.preprocessor.split_dataset(
            images, labels, train_ratio=0.7, val_ratio=0.15
        )

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

        total = (
            len(splits["train"]["images"])
            + len(splits["val"]["images"])
            + len(splits["test"]["images"])
        )
        assert total == n_samples


class TestTrafficSignDataset:
    """Test class cho TrafficSignDataset."""

    def test_dataset_length(self):
        """Test length của dataset."""
        images = np.random.randint(0, 255, (20, 64, 64, 3), dtype=np.uint8)
        labels = np.arange(20) % 5
        dataset = TrafficSignDataset(images, labels)
        assert len(dataset) == 20

    def test_dataset_getitem(self):
        """Test getitem."""
        images = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
        labels = np.arange(10) % 3
        dataset = TrafficSignDataset(images, labels)

        image, label = dataset[0]
        assert image.shape[0] == 3  # C, H, W (tensor format)
        assert isinstance(label, (int, np.integer))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
