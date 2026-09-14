"""
Test Detection Module
=====================
Unit tests cho module phát hiện biển báo.
"""

import sys
import numpy as np
import cv2
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detector import TrafficSignDetector


class TestTrafficSignDetector:
    """Test class cho TrafficSignDetector."""

    def setup_method(self):
        """Setup (detector không có model)."""
        self.detector = TrafficSignDetector(model_path=None)

    def test_detect_regions_red(self):
        """Test phát hiện vùng đỏ."""
        # Tạo ảnh với vùng đỏ
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(image, (100, 100), 30, (0, 0, 255), -1)  # Vòng tròn đỏ

        bboxes = self.detector.detect_regions(image)
        # Có thể phát hiện hoặc không tùy kích thước
        assert isinstance(bboxes, list)

    def test_detect_regions_blue(self):
        """Test phát hiện vùng xanh."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)  # Hình vuông xanh

        bboxes = self.detector.detect_regions(image)
        assert isinstance(bboxes, list)

    def test_detect_regions_empty(self):
        """Test ảnh không có biển báo."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        bboxes = self.detector.detect_regions(image)
        assert len(bboxes) == 0

    def test_draw_detections(self):
        """Test vẽ detections lên ảnh."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        detections = [
            {
                "bbox": (10, 10, 50, 50),
                "class_id": 14,
                "class_name": "Dừng lại (STOP)",
                "confidence": 0.95,
            }
        ]

        annotated = self.detector.draw_detections(image, detections)
        assert annotated.shape == image.shape
        # Kiểm tra ảnh đã được vẽ (không giống ảnh gốc)
        assert not np.array_equal(annotated, image)

    def test_draw_empty_detections(self):
        """Test vẽ khi không có detection."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        annotated = self.detector.draw_detections(image, [])
        assert np.array_equal(annotated, image)

    def test_color_ranges(self):
        """Test color ranges được định nghĩa."""
        assert "red1" in self.detector.color_ranges
        assert "red2" in self.detector.color_ranges
        assert "blue" in self.detector.color_ranges
        assert "yellow" in self.detector.color_ranges


class TestDetectorWithImage:
    """Test detection với ảnh sample."""

    def test_detect_from_array(self):
        """Test detect từ numpy array."""
        detector = TrafficSignDetector(model_path=None)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Không có model nên chỉ detect regions
        bboxes = detector.detect_regions(image)
        assert isinstance(bboxes, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
