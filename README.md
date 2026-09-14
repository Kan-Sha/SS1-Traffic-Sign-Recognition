# 🚦 Hệ Thống Nhận Dạng Biển Báo Giao Thông

## Giới thiệu

Dự án sử dụng **Deep Learning** (CNN + Transfer Learning) để nhận dạng biển báo giao thông từ ảnh và video real-time. Dự án thuộc bộ môn **Chuyên Đề 1 (SS1) **.

### Tính năng chính
- 📸 Nhận dạng biển báo từ ảnh tĩnh (43 loại)
- 🎥 Nhận dạng real-time từ webcam
- 🌐 Giao diện web Streamlit trực quan
- 📊 Thống kê & visualization kết quả

## Cài đặt

### 1. Clone repository
```bash
git clone <repo-url>
cd project_ipr
```

### 2. Tạo virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Tải dataset
```bash
python -m src.data.download_dataset
```

## Sử dụng

### Huấn luyện model
```bash
python -m src.models.train --model resnet18 --epochs 30
```

### Nhận dạng real-time
```bash
python -m src.detection.realtime
```

### Chạy web app
```bash
streamlit run app/app.py
```

### Chạy tests
```bash
python -m pytest tests/ -v
```

## Cấu trúc thư mục

```
project_ipr/
├── config/          # Cấu hình
├── data/            # Dữ liệu
├── src/             # Source code
│   ├── data/        # Data pipeline
│   ├── models/      # Model architecture & training
│   ├── detection/   # Detection & real-time
│   └── utils/       # Utilities
├── app/             # Streamlit web app
├── notebooks/       # Jupyter notebooks
├── tests/           # Unit tests
├── models/          # Saved models
└── reports/         # Báo cáo & figures
```

## Kỹ thuật sử dụng

| Kỹ thuật | Mô tả |
|---------|-------|
| CLAHE | Cân bằng histogram |
| Color Segmentation | Phân đoạn biển báo theo màu |
| Contour Detection | Phát hiện đường viền |
| CNN | Mạng nơ-ron tích chập |
| Transfer Learning | Fine-tune ResNet pretrained |
| GradCAM | Visualization attention map |

## Nhóm thực hiện

| Thành viên | Vai trò |
|-----------|---------|
| TV1 | Data Engineer - Thu thập & xử lý dữ liệu<br><br>ML Engineer - Thiết kế & huấn luyện model |
| TV2 | CV Engineer - Detection & Real-time<br><br> Full-stack - Web App & Tích hợp |
