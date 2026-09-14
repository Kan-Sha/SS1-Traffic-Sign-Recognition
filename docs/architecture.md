# 🏗️ Kiến Trúc Hệ Thống - Nhận Dạng Biển Báo Giao Thông

## 1. Tổng Quan Kiến Trúc

```
┌──────────────────────────────────────────────────────────────────────┐
│                     🌐 STREAMLIT WEB APP                             │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌────────┐ ┌─────────────┐  │
│  │  Home    │ │  Upload  │ │Detection│ │GradCAM │ │  Statistics  │  │
│  │  Page    │ │  & Nhận  │ │Pipeline │ │Visuali-│ │  & Đánh Giá │  │
│  │         │ │  Dạng    │ │         │ │zation  │ │             │  │
│  └──────────┘ └────┬─────┘ └────┬────┘ └───┬────┘ └──────┬──────┘  │
└────────────────────┼────────────┼──────────┼─────────────┼──────────┘
                     │            │          │             │
         ┌───────────┼────────────┼──────────┼─────────────┘
         ▼           ▼            ▼          ▼
┌──────────────────────────────────────────────────────────┐
│                   🔧 CORE MODULES                         │
│                                                           │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐ │
│  │   Detector   │  │  Classifier │  │    Evaluator     │ │
│  │  (OpenCV)    │  │  (PyTorch)  │  │  (sklearn+torch) │ │
│  │              │  │             │  │                  │ │
│  │  • HSV Color │  │  • CNN      │  │  • Metrics       │ │
│  │  • Contour   │  │  • ResNet18 │  │  • Confusion Mat │ │
│  │  • Morphology│  │  • Softmax  │  │  • Error Analysis│ │
│  │  • BBox      │  │  • Top-5    │  │  • Per-class Acc │ │
│  └──────┬───────┘  └──────┬──────┘  └────────┬─────────┘ │
│         │                 │                  │           │
│  ┌──────┴─────────────────┴──────────────────┘           │
│  ▼                                                       │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐ │
│  │  Real-time   │  │Visualization│  │ Report Generator │ │
│  │  (Webcam)    │  │(GradCAM,    │  │ (Markdown/JSON)  │ │
│  │              │  │ Charts)     │  │                  │ │
│  └──────────────┘  └─────────────┘  └──────────────────┘ │
└──────────────────────────┬───────────────────────────────┘
                           │
           ┌───────────────┼───────────────────┐
           ▼               ▼                   ▼
┌──────────────────────────────────────────────────────────┐
│                   📁 DATA PIPELINE                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
│  │ Download     │ │ Preprocess   │ │  Augmentation    │ │
│  │ (GTSRB)     │ │ • CLAHE      │ │  • Rotation      │ │
│  │ • Train     │ │ • Resize 64² │ │  • Color Jitter  │ │
│  │ • Test      │ │ • Normalize  │ │  • Noise/Blur    │ │
│  │ • Labels    │ │ • Split Data │ │  • Perspective   │ │
│  └──────────────┘ └──────────────┘ └──────────────────┘ │
└──────────────────────────┬───────────────────────────────┘
                           ▼
              ┌──────────────────────────┐
              │  📦 GTSRB Dataset        │
              │  43 classes · ~39K imgs  │
              │  Train/Val/Test split    │
              └──────────────────────────┘
```

## 2. Cấu Trúc Thư Mục

```
project_ipr/
├── app/
│   ├── app.py                  # Streamlit App (5 trang)
│   └── assets/
│       └── style.css           # Premium Dark Theme CSS
├── config/
│   └── config.yaml             # Cấu hình tập trung
├── data/
│   ├── raw/                    # GTSRB gốc (zip + extracted)
│   ├── processed/              # Numpy arrays (.npy)
│   └── samples/                # Ảnh mẫu
├── docs/
│   ├── architecture.md         # Tài liệu kiến trúc (file này)
│   └── report.md               # Báo cáo đánh giá model
├── models/
│   └── best_model.pth          # Model đã train (~139MB)
├── reports/
│   └── figures/                # Biểu đồ evaluation
│       ├── confusion_matrix.png
│       ├── per_class_accuracy.png
│       ├── confidence_distribution.png
│       ├── training_curves.png
│       ├── confused_pairs.png
│       ├── evaluation_metrics.json
│       └── classification_report.txt
├── src/
│   ├── data/
│   │   ├── download_dataset.py # Tải GTSRB dataset
│   │   ├── preprocessing.py    # CLAHE + Resize + Normalize
│   │   └── augmentation.py     # Data augmentation
│   ├── models/
│   │   ├── cnn_model.py        # Custom CNN architecture
│   │   ├── transfer_model.py   # ResNet18/50 Transfer Learning
│   │   └── train.py            # Training pipeline
│   ├── detection/
│   │   ├── detector.py         # Color Seg + Classification
│   │   └── realtime.py         # Webcam real-time detection
│   ├── evaluation/
│   │   ├── evaluate.py         # Model evaluation (metrics + charts)
│   │   └── report_generator.py # Auto-generate Markdown report
│   └── utils/
│       ├── helpers.py          # Config, device, model loading
│       └── visualization.py    # GradCAM, training curves
├── tests/
│   ├── test_detection.py
│   ├── test_model.py
│   └── test_preprocessing.py
├── requirements.txt
└── README.md
```

## 3. Luồng Xử Lý Chi Tiết

### 3.1 Training Pipeline

```
📥 GTSRB Download & Extract
        │
        ▼
┌─────────────────────────────┐
│ Preprocessing               │
│ ┌─────────┐ ┌─────────────┐│
│ │ CLAHE   │→│ Resize 64²  ││
│ └─────────┘ └──────┬──────┘│
│              ┌─────┴──────┐│
│              │ Normalize  ││
│              │ μ=[.34,.31]││
│              │ σ=[.27,.26]││
│              └─────┬──────┘│
└────────────────────┼───────┘
                     │
        ┌────────────┴───────────┐
        ▼                        ▼
┌──────────────┐      ┌──────────────────┐
│ Data Split   │      │ Augmentation     │
│ Train: 70%   │      │ • RandomRotation │
│ Val:   15%   │      │ • ColorJitter    │
│ Test:  15%   │      │ • RandomAffine   │
└──────┬───────┘      │ • GaussianNoise  │
       │              │ • Perspective    │
       ▼              └────────┬─────────┘
┌──────────────┐               │
│ DataLoader   │◄──────────────┘
│ batch=64     │
│ shuffle=True │
│ pin_memory   │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────┐
│           Training Loop                  │
│                                          │
│  ┌──────────┐    ┌──────────────┐       │
│  │ ResNet18 │    │   Optimizer  │       │
│  │(pretrain)│    │  Adam        │       │
│  │          │    │  lr=0.001    │       │
│  │ layer1-4 │    │  wd=1e-4    │       │
│  │    ↓     │    └──────────────┘       │
│  │ Classifier│    ┌──────────────┐      │
│  │ 512→512  │    │   Scheduler  │       │
│  │ 512→256  │    │  StepLR      │       │
│  │ 256→43   │    │  step=10     │       │
│  └──────────┘    │  γ=0.1      │       │
│                  └──────────────┘       │
│  ┌──────────────────────────────┐       │
│  │ CrossEntropyLoss             │       │
│  │ + Early Stopping (patience=7)│       │
│  │ + Checkpointing (best_model) │       │
│  └──────────────────────────────┘       │
└─────────────────────┬───────────────────┘
                      │
                      ▼
              ┌──────────────┐
              │ best_model   │
              │   .pth       │
              │ (139MB)      │
              └──────────────┘
```

### 3.2 Detection Pipeline

```
Input Image (BGR)
       │
       ▼
┌─────────────────────────────────────┐
│ Step 1: Color Segmentation (HSV)    │
│                                      │
│  BGR → HSV                           │
│  ┌───────────┐ ┌──────┐ ┌────────┐ │
│  │Red Mask   │ │Blue  │ │Yellow  │ │
│  │H:[0-10]   │ │Mask  │ │Mask    │ │
│  │H:[170-180]│ │H:100 │ │H:15-35│ │
│  │S:[70-255] │ │-130  │ │        │ │
│  └─────┬─────┘ └──┬───┘ └───┬────┘ │
│        └───────┬───┘         │      │
│                ▼             │      │
│         Combined Mask ◄──────┘      │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ Step 2: Morphological Operations    │
│  MORPH_CLOSE (kernel=5, iter=2)     │
│  MORPH_OPEN  (kernel=5, iter=1)     │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ Step 3: Contour Detection           │
│  findContours(RETR_EXTERNAL)        │
│  Filter: area ∈ [500, 50000]        │
│  Filter: aspect_ratio ∈ [0.5, 2.0]  │
│  Output: Bounding Boxes + padding   │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ Step 4: Classification              │
│  Crop → Resize(64²) → Normalize    │
│  → ResNet18 → Softmax → Top-5      │
│  Filter: confidence ≥ threshold     │
└────────────────┬────────────────────┘
                 │
                 ▼
         Detection Results
         [class_name, confidence,
          bbox, top5_predictions]
```

### 3.3 Evaluation Pipeline

```
best_model.pth + Test Set (5,882 images)
                 │
                 ▼
┌─────────────────────────────────────────┐
│ ModelEvaluator                           │
│                                          │
│  Forward pass (no_grad) trên test set   │
│  → predictions, probabilities, labels    │
│                                          │
│  ┌──────────────────────────────┐       │
│  │ Overall Metrics              │       │
│  │ • Top-1 Accuracy: 99.88%    │       │
│  │ • Top-5 Accuracy: 99.98%    │       │
│  │ • F1 Macro: 0.9994          │       │
│  │ • Precision: 0.9994         │       │
│  │ • Recall: 0.9993            │       │
│  └──────────────────────────────┘       │
│                                          │
│  ┌──────────────────────────────┐       │
│  │ Error Analysis               │       │
│  │ • 7 errors / 5882 samples    │       │
│  │ • Avg correct conf: 0.9992   │       │
│  │ • Avg error conf: 0.7363     │       │
│  │ • Worst: speed limit signs   │       │
│  └──────────────────────────────┘       │
│                                          │
│  Outputs:                                │
│  • confusion_matrix.png                  │
│  • per_class_accuracy.png                │
│  • confidence_distribution.png           │
│  • training_curves.png                   │
│  • confused_pairs.png                    │
│  • evaluation_metrics.json               │
│  • classification_report.txt             │
│  • docs/report.md (auto-generated)       │
└─────────────────────────────────────────┘
```

### 3.4 GradCAM Pipeline

```
Input Image + Trained Model
         │
         ▼
┌──────────────────────────────────┐
│ Preprocess: Resize → Normalize   │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│ Forward Hook: layer4[-1]         │
│ → Store activations (A)          │
│                                  │
│ Backward Hook: layer4[-1]        │
│ → Store gradients (∂y/∂A)        │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│ GradCAM Computation              │
│                                  │
│ weights = GAP(gradients)         │
│ cam = ReLU(Σ wₖ · Aₖ)           │
│ cam = normalize(cam) → [0, 1]   │
│ heatmap = colormap(cam)          │
│ overlay = 0.6·image + 0.4·heat  │
└──────────────────────────────────┘
```

## 4. Model Architecture

### ResNet18 + Custom Classifier

```
Input: (B, 3, 64, 64)
        │
        ▼
┌──────────────────────────┐
│ ResNet18 Backbone         │
│ (pretrained ImageNet)     │
│                           │
│ conv1: 7×7, 64, stride=2 │
│ bn1 → relu → maxpool     │
│                           │
│ layer1: BasicBlock × 2    │
│   [3×3, 64] × 2          │
│                           │
│ layer2: BasicBlock × 2    │
│   [3×3, 128] × 2         │
│                           │
│ layer3: BasicBlock × 2    │
│   [3×3, 256] × 2         │
│                           │
│ layer4: BasicBlock × 2    │ ← GradCAM target
│   [3×3, 512] × 2         │
│                           │
│ AdaptiveAvgPool2d → (512) │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Custom Classifier         │
│                           │
│ Linear(512 → 512)         │
│ BatchNorm1d(512)          │
│ ReLU + Dropout(0.5)       │
│                           │
│ Linear(512 → 256)         │
│ BatchNorm1d(256)          │
│ ReLU + Dropout(0.25)      │
│                           │
│ Linear(256 → 43)          │
└────────────┬─────────────┘
             │
             ▼
       Output: (B, 43)
       → Softmax → Predictions
```

## 5. Streamlit Web App

```
┌──────────────────────────────────────────────────────────┐
│ 🚦 Traffic Sign AI - Streamlit App                       │
│                                                           │
│ ┌─────────────┐  ┌──────────────────────────────────────┐│
│ │  Sidebar    │  │  Main Content                        ││
│ │             │  │                                      ││
│ │ 🏠 Home     │  │  Page 1: Overview + Feature Cards    ││
│ │ 📸 Upload   │  │          + Metrics + Team Info       ││
│ │ 🔍 Detect   │  │                                      ││
│ │ 🧠 GradCAM  │  │  Page 2: Upload → Classify          ││
│ │ 📊 Stats    │  │          → Top-5 Predictions        ││
│ │             │  │                                      ││
│ │ ┌─────────┐│  │  Page 3: Upload → Detect Regions    ││
│ │ │ Model   ││  │          → HSV Masks → Annotate     ││
│ │ │ Status  ││  │                                      ││
│ │ │ ResNet18││  │  Page 4: Upload → GradCAM Heatmap   ││
│ │ │ 99.88%  ││  │          → Overlay → Explanation    ││
│ │ └─────────┘│  │                                      ││
│ │             │  │  Page 5: Metrics + Training Curves  ││
│ │ GTSRB·43cls│  │          + Confusion Matrix + Tabs  ││
│ └─────────────┘  └──────────────────────────────────────┘│
└──────────────────────────────────────────────────────────┘
```

## 6. Công Nghệ Sử Dụng

| Thành phần | Công nghệ |
|-----------|-----------|
| Deep Learning | PyTorch + torchvision |
| Computer Vision | OpenCV |
| Web App | Streamlit |
| Data | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Evaluation | scikit-learn |
| Model | ResNet18 (Transfer Learning) |
| Dataset | GTSRB (43 classes, ~39K images) |
