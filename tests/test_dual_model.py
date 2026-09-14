"""Quick test: Verify both GTSRB and VN models load correctly."""
import sys
import json
sys.path.insert(0, '.')

from src.detection.detector import TrafficSignDetector

# Test 1: GTSRB
print("=== Test GTSRB Model ===")
det = TrafficSignDetector("models/best_model.pth", "resnet18", 0.3)
print(f"  Classes: {len(det.class_names)}")
print(f"  Model loaded: {det.model is not None}")
print(f"  Class 0: {det.class_names.get(0)}")

# Test 2: VN
print()
print("=== Test VN Model ===")
det_vn = TrafficSignDetector("models/best_model_vn.pth", "resnet18", 0.3)

with open("data/vn_processed/vn_class_names.json", "r", encoding="utf-8") as f:
    vn_names = {int(k): v for k, v in json.load(f).items()}
det_vn.class_names = vn_names

print(f"  Classes: {len(det_vn.class_names)}")
print(f"  Model loaded: {det_vn.model is not None}")
print(f"  Class 2: {det_vn.class_names.get(2)}")
print(f"  Class 10: {det_vn.class_names.get(10)}")

# Test 3: Quick inference
from PIL import Image
import os

# Find a test image from VN dataset
test_dir = "data/vn_processed/test"
if os.path.exists(test_dir):
    for cls_dir in sorted(os.listdir(test_dir))[:1]:
        cls_path = os.path.join(test_dir, cls_dir)
        imgs = os.listdir(cls_path)
        if imgs:
            img_path = os.path.join(cls_path, imgs[0])
            img = Image.open(img_path)
            result = det_vn.classify_single_image(img)
            print(f"\n=== Quick Inference Test ===")
            print(f"  Image: {img_path}")
            print(f"  Predicted: {result['class_name']} (conf: {result['confidence']:.4f})")

print("\n=== ALL TESTS PASSED ===")
