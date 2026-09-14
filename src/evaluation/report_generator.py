"""
Report Generator
=================
Tự động sinh báo cáo tóm tắt Markdown từ kết quả evaluation.

Sử dụng:
    python -m src.evaluation.report_generator
"""

import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download_dataset import CLASS_NAMES


def generate_report(metrics_path="reports/figures/evaluation_metrics.json",
                    output_path="docs/report.md",
                    figures_dir="reports/figures"):
    """
    Tạo báo cáo tóm tắt Markdown từ kết quả evaluation.

    Args:
        metrics_path: Đường dẫn file JSON metrics
        output_path: Đường dẫn file output
        figures_dir: Thư mục chứa biểu đồ
    """
    metrics_path = Path(metrics_path)
    output_path = Path(output_path)
    figures_dir = Path(figures_dir)

    if not metrics_path.exists():
        print(f"❌ Không tìm thấy file metrics: {metrics_path}")
        print("   Chạy evaluation trước: python -m src.evaluation.evaluate")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    overall = metrics["overall"]
    error_analysis = metrics["error_analysis"]
    per_class = metrics["per_class"]
    confused = metrics.get("confused_pairs", [])
    model_info = metrics.get("model_info", {})

    # Tìm best/worst classes
    classes_sorted = sorted(per_class.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    best_5 = classes_sorted[:5]
    worst_5 = classes_sorted[-5:][::-1]

    # Relative path for figures
    rel_figures = Path("..") / figures_dir

    report = f"""# 📊 Báo Cáo Đánh Giá Model - Nhận Dạng Biển Báo Giao Thông

> **Ngày tạo:** {datetime.now().strftime("%d/%m/%Y %H:%M")}
> **Model:** ResNet18 + Transfer Learning
> **Dataset:** GTSRB (German Traffic Sign Recognition Benchmark)

---

## 1. Tổng Quan Dự Án

Dự án xây dựng hệ thống nhận dạng biển báo giao thông sử dụng **Deep Learning**, phục vụ cho mục tiêu **an toàn giao thông**. Hệ thống có khả năng:

- Nhận dạng **43 loại biển báo** từ ảnh tĩnh
- Phát hiện và nhận dạng biển báo **real-time** từ webcam
- Giao diện web **Streamlit** trực quan

### Kỹ thuật IPR sử dụng

| Kỹ thuật | Mục đích |
|----------|---------|
| CLAHE | Cân bằng histogram thích ứng - tăng cường contrast |
| Color Segmentation (HSV) | Phân đoạn vùng biển báo theo màu |
| Contour Detection | Phát hiện đường viền biển báo |
| Morphological Operations | Loại bỏ nhiễu (erosion, dilation) |
| CNN (Convolutional Neural Network) | Trích xuất đặc trưng từ ảnh |
| Transfer Learning (ResNet18) | Fine-tune model pretrained ImageNet |
| Data Augmentation | Tăng cường dữ liệu training |
| GradCAM | Visualization vùng model chú ý |

---

## 2. Kết Quả Đánh Giá

### 2.1 Overall Metrics

| Metric | Giá trị |
|--------|---------|
| **Top-1 Accuracy** | **{overall['top1_accuracy']:.2f}%** |
| **Top-5 Accuracy** | **{overall['top5_accuracy']:.2f}%** |
| Precision (Macro) | {overall['precision_macro']:.4f} |
| Recall (Macro) | {overall['recall_macro']:.4f} |
| F1-Score (Macro) | {overall['f1_macro']:.4f} |
| F1-Score (Weighted) | {overall['f1_weighted']:.4f} |
| Tổng mẫu test | {overall['total_samples']} |

### 2.2 Phân Tích Lỗi

| Thống kê | Giá trị |
|----------|---------|
| Tổng số lỗi | {error_analysis['total_errors']} |
| Tỷ lệ lỗi | {error_analysis['error_rate']:.2f}% |
| Confidence TB (dự đoán đúng) | {error_analysis['avg_correct_confidence']:.4f} |
| Confidence TB (dự đoán sai) | {error_analysis['avg_error_confidence']:.4f} |

### 2.3 Training Curves

![Training Curves]({rel_figures}/training_curves.png)

### 2.4 Confusion Matrix

![Confusion Matrix]({rel_figures}/confusion_matrix.png)

### 2.5 Per-class Accuracy

![Per-class Accuracy]({rel_figures}/per_class_accuracy.png)

### 2.6 Confidence Distribution

![Confidence Distribution]({rel_figures}/confidence_distribution.png)

---

## 3. Phân Tích Chi Tiết

### 3.1 Top 5 Classes Có Accuracy Cao Nhất

| Class ID | Tên biển báo | Accuracy | Số mẫu |
|----------|-------------|----------|--------|
"""

    for class_id, info in best_5:
        report += f"| {class_id} | {info['name']} | {info['accuracy']:.1f}% | {info['support']} |\n"

    report += f"""
### 3.2 Top 5 Classes Có Accuracy Thấp Nhất

| Class ID | Tên biển báo | Accuracy | Số mẫu |
|----------|-------------|----------|--------|
"""

    for class_id, info in worst_5:
        report += f"| {class_id} | {info['name']} | {info['accuracy']:.1f}% | {info['support']} |\n"

    report += f"""
### 3.3 Top Cặp Class Hay Nhầm Lẫn

| True Label | Predicted Label | Số lần |
|-----------|----------------|--------|
"""

    for pair in confused[:7]:
        report += f"| {pair['true']} | {pair['pred']} | {pair['count']} |\n"

    report += f"""
![Confused Pairs]({rel_figures}/confused_pairs.png)

---

## 4. Kiến Trúc Model

### 4.1 ResNet18 + Transfer Learning

```
ResNet18 (pretrained ImageNet)
├── conv1 (7x7, 64)
├── layer1 (BasicBlock x2, 64)
├── layer2 (BasicBlock x2, 128)
├── layer3 (BasicBlock x2, 256)
├── layer4 (BasicBlock x2, 512)
├── Global Average Pooling
└── Custom Classifier:
    ├── Linear(512 → 512) + BN + ReLU + Dropout(0.5)
    ├── Linear(512 → 256) + BN + ReLU + Dropout(0.25)
    └── Linear(256 → 43)
```

### 4.2 Detection Pipeline

```
Input Image
    │
    ▼
HSV Color Segmentation (Red, Blue, Yellow)
    │
    ▼
Morphological Operations (Close + Open)
    │
    ▼
Contour Detection + Bounding Box
    │
    ▼
Crop Region → Resize (64x64) → Normalize
    │
    ▼
ResNet18 Classification → Top-5 Predictions
```

---

## 5. Cấu Hình Training

| Tham số | Giá trị |
|---------|---------|
| Image Size | 64 × 64 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| LR Scheduler | StepLR (step=10, gamma=0.1) |
| Early Stopping | patience=7 |
| Augmentation | Medium (rotation, affine, color jitter, perspective) |
| Train/Val/Test | 70% / 15% / 15% |

---

## 6. Kết Luận

- Model đạt **Top-1 Accuracy {overall['top1_accuracy']:.2f}%** và **Top-5 Accuracy {overall['top5_accuracy']:.2f}%** trên test set
- Confidence trung bình cho dự đoán đúng ({error_analysis['avg_correct_confidence']:.4f}) cao hơn đáng kể so với dự đoán sai ({error_analysis['avg_error_confidence']:.4f})
- Hệ thống có thể được triển khai cho ứng dụng nhận dạng biển báo thực tế
- Các class có accuracy thấp cần thêm dữ liệu augmentation hoặc specialized training

---

*Báo cáo được tạo tự động bởi `src/evaluation/report_generator.py`*
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ Đã tạo báo cáo: {output_path}")
    return report


if __name__ == "__main__":
    generate_report()
