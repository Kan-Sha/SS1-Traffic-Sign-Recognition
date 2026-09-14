"""
Traffic Sign Detector
====================
Phát hiện và nhận dạng biển báo giao thông trong ảnh.
Sử dụng color segmentation + contour detection để phát hiện vùng biển báo,
sau đó sử dụng CNN/ResNet để phân loại.

Sử dụng:
    from src.detection.detector import TrafficSignDetector
    detector = TrafficSignDetector("models/best_model.pth")
    results = detector.detect("path/to/image.jpg")
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms


# --- Font cache (load once, reuse forever) ---
_font_cache = {}
_dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))


def _get_unicode_font(size=16):
    """Load a Unicode-capable font for Vietnamese text rendering (cached)."""
    if size in _font_cache:
        return _font_cache[size]

    font_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for font_path in font_candidates:
        try:
            font = ImageFont.truetype(font_path, size)
            _font_cache[size] = font
            return font
        except (IOError, OSError):
            continue
    font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def _put_unicode_text(img_bgr, text, position, font_size=16, color=(255, 255, 255)):
    """
    Draw Unicode text on an OpenCV BGR image using PIL.

    Args:
        img_bgr: numpy array (BGR)
        text: Unicode string to draw
        position: (x, y) tuple
        font_size: font size in pixels
        color: BGR color tuple

    Returns:
        numpy array (BGR) with text drawn
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_unicode_font(font_size)
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _put_unicode_text_pil(draw, text, position, font_size=16, color=(255, 255, 255)):
    """
    Draw Unicode text directly on an existing PIL ImageDraw context.
    Much faster than _put_unicode_text when drawing multiple texts.
    Color is BGR format for compatibility.
    """
    font = _get_unicode_font(font_size)
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)


def _get_text_size_unicode(text, font_size=16):
    """Get the pixel size of Unicode text (cached font)."""
    font = _get_unicode_font(font_size)
    bbox = _dummy_draw.textbbox((0, 0), text, font=font)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


class TrafficSignDetector:
    """
    Phát hiện và nhận dạng biển báo giao thông.

    Pipeline:
        1. Phân đoạn màu (HSV) → Tìm vùng đỏ, xanh, vàng
        2. Morphological operations → Loại bỏ nhiễu
        3. Contour detection → Tìm đường viền biển báo
        4. Bounding box → Crop vùng biển báo
        5. Classification → Phân loại biển báo bằng CNN/ResNet

    Args:
        model_path: Đường dẫn file model (.pth)
        model_type: Loại model ("custom_cnn", "resnet18", "resnet50")
        confidence_threshold: Ngưỡng tin cậy tối thiểu
        device: Device (cuda/cpu)
    """

    def __init__(self, model_path=None, model_type="resnet18",
                 confidence_threshold=0.7, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.model = None

        # Transform cho classification
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3403, 0.3121, 0.3214],
                std=[0.2724, 0.2608, 0.2669],
            ),
        ])

        # HSV color ranges cho biển báo
        self.color_ranges = {
            "red1": (np.array([0, 70, 50]), np.array([10, 255, 255])),
            "red2": (np.array([170, 70, 50]), np.array([180, 255, 255])),
            "blue": (np.array([100, 70, 50]), np.array([130, 255, 255])),
            "yellow": (np.array([15, 70, 50]), np.array([35, 255, 255])),
        }

        # Min/Max area cho contour
        self.min_area = 500
        self.max_area = 50000

        # Load model nếu có
        if model_path:
            self.load_model(model_path, model_type)

        # Class names
        from src.data.download_dataset import CLASS_NAMES
        self.class_names = CLASS_NAMES

    def load_model(self, model_path, model_type="resnet18"):
        """
        Load model đã train.

        Args:
            model_path: Đường dẫn file model
            model_type: Loại model
        """
        from src.models.transfer_model import create_model

        # Auto-detect num_classes từ checkpoint
        checkpoint = torch.load(str(model_path), map_location=self.device)
        num_classes = checkpoint.get("num_classes", 43)

        self.model = create_model(
            model_type=model_type,
            num_classes=num_classes,
            pretrained=False,
        )

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[OK] Model loaded: {model_path}")

    def detect_regions(self, image):
        """
        Phát hiện vùng chứa biển báo bằng color segmentation.

        Args:
            image: numpy array ảnh BGR

        Returns:
            List of bounding boxes [(x, y, w, h), ...]
        """
        # Chuyển sang HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Tạo mask cho các màu biển báo
        masks = []
        for name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, lower, upper)
            masks.append(mask)

        # Kết hợp tất cả mask
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Tìm contours
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Lọc contours theo diện tích và tỷ lệ
        bounding_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h

                # Biển báo thường có tỷ lệ gần vuông
                if 0.5 < aspect_ratio < 2.0:
                    # Mở rộng bounding box một chút
                    pad = int(max(w, h) * 0.1)
                    x = max(0, x - pad)
                    y = max(0, y - pad)
                    w = min(image.shape[1] - x, w + 2 * pad)
                    h = min(image.shape[0] - y, h + 2 * pad)
                    bounding_boxes.append((x, y, w, h))

        return bounding_boxes

    def classify_region(self, image, bbox):
        """
        Phân loại vùng biển báo đã crop.

        Args:
            image: numpy array ảnh BGR gốc
            bbox: tuple (x, y, w, h) bounding box

        Returns:
            Dict: {class_id, class_name, confidence, top5}
        """
        if self.model is None:
            return None

        x, y, w, h = bbox
        # Crop vùng biển báo
        crop = image[y:y + h, x:x + w]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(crop_rgb)

        # Transform và predict
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)

        # Top-5 predictions
        top5_probs, top5_indices = torch.topk(probs, 5)
        top5_probs = top5_probs.squeeze().cpu().numpy()
        top5_indices = top5_indices.squeeze().cpu().numpy()

        class_id = int(top5_indices[0])
        confidence = float(top5_probs[0])

        return {
            "class_id": class_id,
            "class_name": self.class_names.get(class_id, f"Class {class_id}"),
            "confidence": confidence,
            "top5": [
                {
                    "class_id": int(idx),
                    "class_name": self.class_names.get(int(idx), f"Class {idx}"),
                    "confidence": float(prob),
                }
                for idx, prob in zip(top5_indices, top5_probs)
            ],
        }

    def classify_single_image(self, pil_image):
        """
        Phân loại một ảnh biển báo đơn lẻ (PIL Image).

        Args:
            pil_image: PIL Image (RGB)

        Returns:
            Dict: {class_id, class_name, confidence, top5}
        """
        if self.model is None:
            raise RuntimeError("Model chưa được load!")

        if not isinstance(pil_image, Image.Image):
            pil_image = Image.fromarray(pil_image)
        pil_image = pil_image.convert("RGB")

        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)

        top5_probs, top5_indices = torch.topk(probs, 5)
        top5_probs = top5_probs.squeeze().cpu().numpy()
        top5_indices = top5_indices.squeeze().cpu().numpy()

        class_id = int(top5_indices[0])
        confidence = float(top5_probs[0])

        return {
            "class_id": class_id,
            "class_name": self.class_names.get(class_id, f"Class {class_id}"),
            "confidence": confidence,
            "top5": [
                {
                    "class_id": int(idx),
                    "class_name": self.class_names.get(int(idx), f"Class {idx}"),
                    "confidence": float(prob),
                }
                for idx, prob in zip(top5_indices, top5_probs)
            ],
        }

    def detect(self, image_input, downscale_for_detection=None):
        """
        Pipeline phát hiện & nhận dạng biển báo.

        Args:
            image_input: Đường dẫn ảnh hoặc numpy array
            downscale_for_detection: Nếu set (vd 640), sẽ downscale frame
                để chạy color segmentation nhanh hơn, rồi scale bbox lại.

        Returns:
            List of detection results
        """
        # Load ảnh
        if isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input))
            if image is None:
                raise FileNotFoundError(f"Không thể đọc ảnh: {image_input}")
        else:
            image = image_input  # No copy needed — detect_regions is read-only

        # Phát hiện vùng biển báo (optional downscale for speed)
        if downscale_for_detection and image.shape[1] > downscale_for_detection:
            scale = downscale_for_detection / image.shape[1]
            small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            small_bboxes = self.detect_regions(small)
            # Scale bounding boxes back to original size
            inv_scale = 1.0 / scale
            bounding_boxes = [
                (int(x * inv_scale), int(y * inv_scale),
                 int(w * inv_scale), int(h * inv_scale))
                for x, y, w, h in small_bboxes
            ]
        else:
            bounding_boxes = self.detect_regions(image)

        # Phân loại từng vùng
        results = []
        for bbox in bounding_boxes:
            classification = self.classify_region(image, bbox)

            if classification and classification["confidence"] >= self.confidence_threshold:
                results.append({
                    "bbox": bbox,
                    **classification,
                })

        return results

    def draw_detections(self, image, detections):
        """
        Vẽ bounding box và label lên ảnh.
        Optimized: single BGR→RGB→BGR conversion for all text draws.

        Args:
            image: numpy array ảnh BGR
            detections: List kết quả từ detect()

        Returns:
            Ảnh đã vẽ annotations
        """
        annotated = image.copy()

        # Phase 1: Draw all OpenCV primitives (rectangles) — fast
        text_jobs = []  # Collect text jobs for batch PIL rendering
        for det in detections:
            x, y, w, h = det["bbox"]
            confidence = det["confidence"]
            class_name = det["class_name"]

            # Màu bounding box
            color = (0, 255, 0)  # Green
            if confidence < 0.8:
                color = (0, 255, 255)  # Yellow
            if confidence < 0.6:
                color = (0, 0, 255)  # Red

            # Vẽ bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Chuẩn bị label
            label = f"{class_name} ({confidence:.0%})"
            font_size = 16
            label_w, label_h = _get_text_size_unicode(label, font_size)

            # Nền cho label
            cv2.rectangle(
                annotated,
                (x, y - label_h - 10),
                (x + label_w + 10, y),
                color,
                -1,
            )

            text_jobs.append((label, (x + 5, y - label_h - 5), font_size, (0, 0, 0)))

        # Phase 2: Batch render all Unicode text in ONE PIL session
        if text_jobs:
            img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)
            for text, pos, fsize, color in text_jobs:
                _put_unicode_text_pil(draw, text, pos, fsize, color)
            annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return annotated

    def classify_single_image(self, image_input):
        """
        Phân loại 1 ảnh biển báo đã crop sẵn (không cần detection).

        Args:
            image_input: Đường dẫn ảnh hoặc numpy array hoặc PIL Image

        Returns:
            Dict kết quả classification
        """
        if self.model is None:
            raise RuntimeError("Model chưa được load. Gọi load_model() trước.")

        # Chuyển đổi input
        if isinstance(image_input, (str, Path)):
            pil_image = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
        else:
            raise ValueError("Input phải là đường dẫn, numpy array, hoặc PIL Image.")

        # Transform và predict
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)

        top5_probs, top5_indices = torch.topk(probs, 5)
        top5_probs = top5_probs.squeeze().cpu().numpy()
        top5_indices = top5_indices.squeeze().cpu().numpy()

        class_id = int(top5_indices[0])

        return {
            "class_id": class_id,
            "class_name": self.class_names.get(class_id, f"Class {class_id}"),
            "confidence": float(top5_probs[0]),
            "top5": [
                {
                    "class_id": int(idx),
                    "class_name": self.class_names.get(int(idx), f"Class {idx}"),
                    "confidence": float(prob),
                }
                for idx, prob in zip(top5_indices, top5_probs)
            ],
        }


if __name__ == "__main__":
    print("🚦 Traffic Sign Detector")
    print("=" * 40)
    print("Sử dụng:")
    print("  detector = TrafficSignDetector('models/best_model.pth')")
    print("  results = detector.detect('image.jpg')")
    print("  annotated = detector.draw_detections(image, results)")
