# 📊 Báo Cáo Đánh Giá Model - Nhận Dạng Biển Báo Giao Thông

> **Ngày tạo:** 15/04/2026 00:41
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
| **Top-1 Accuracy** | **99.88%** |
| **Top-5 Accuracy** | **99.98%** |
| Precision (Macro) | 0.9994 |
| Recall (Macro) | 0.9993 |
| F1-Score (Macro) | 0.9994 |
| F1-Score (Weighted) | 0.9988 |
| Tổng mẫu test | 5882 |

### 2.2 Phân Tích Lỗi

| Thống kê | Giá trị |
|----------|---------|
| Tổng số lỗi | 7 |
| Tỷ lệ lỗi | 0.12% |
| Confidence TB (dự đoán đúng) | 0.9992 |
| Confidence TB (dự đoán sai) | 0.7363 |

### 2.3 Training Curves

![Training Curves](..\reports\figures/training_curves.png)

### 2.4 Confusion Matrix

![Confusion Matrix](..\reports\figures/confusion_matrix.png)

### 2.5 Per-class Accuracy

![Per-class Accuracy](..\reports\figures/per_class_accuracy.png)

### 2.6 Confidence Distribution

![Confidence Distribution](..\reports\figures/confidence_distribution.png)

---

## 3. Phân Tích Chi Tiết

### 3.1 Top 5 Classes Có Accuracy Cao Nhất

| Class ID | Tên biển báo | Accuracy | Số mẫu |
|----------|-------------|----------|--------|
| 0 | Giới hạn tốc độ (20km/h) | 100.0% | 31 |
| 2 | Giới hạn tốc độ (50km/h) | 100.0% | 338 |
| 4 | Giới hạn tốc độ (70km/h) | 100.0% | 297 |
| 6 | Hết giới hạn tốc độ (80km/h) | 100.0% | 63 |
| 7 | Giới hạn tốc độ (100km/h) | 100.0% | 216 |

### 3.2 Top 5 Classes Có Accuracy Thấp Nhất

| Class ID | Tên biển báo | Accuracy | Số mẫu |
|----------|-------------|----------|--------|
| 5 | Giới hạn tốc độ (80km/h) | 98.9% | 279 |
| 15 | Cấm xe cộ | 99.0% | 95 |
| 1 | Giới hạn tốc độ (30km/h) | 99.4% | 333 |
| 3 | Giới hạn tốc độ (60km/h) | 99.5% | 212 |
| 42 | Hết cấm vượt (xe > 3.5 tấn) | 100.0% | 36 |

### 3.3 Top Cặp Class Hay Nhầm Lẫn

| True Label | Predicted Label | Số lần |
|-----------|----------------|--------|
| [5] Giới hạn tốc độ (80km/h) | [1] Giới hạn tốc độ (30km/h) | 2 |
| [1] Giới hạn tốc độ (30km/h) | [4] Giới hạn tốc độ (70km/h) | 1 |
| [1] Giới hạn tốc độ (30km/h) | [8] Giới hạn tốc độ (120km/h) | 1 |
| [3] Giới hạn tốc độ (60km/h) | [5] Giới hạn tốc độ (80km/h) | 1 |
| [5] Giới hạn tốc độ (80km/h) | [2] Giới hạn tốc độ (50km/h) | 1 |
| [15] Cấm xe cộ | [13] Nhường đường | 1 |

![Confused Pairs](..\reports\figures/confused_pairs.png)

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

- Model đạt **Top-1 Accuracy 99.88%** và **Top-5 Accuracy 99.98%** trên test set
- Confidence trung bình cho dự đoán đúng (0.9992) cao hơn đáng kể so với dự đoán sai (0.7363)
- Hệ thống có thể được triển khai cho ứng dụng nhận dạng biển báo thực tế
- Các class có accuracy thấp cần thêm dữ liệu augmentation hoặc specialized training

---

*Báo cáo được tạo tự động bởi `src/evaluation/report_generator.py`*
