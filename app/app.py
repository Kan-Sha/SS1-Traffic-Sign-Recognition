"""
🚦 Traffic Sign Recognition - Streamlit Web App
================================================
Giao diện web thông minh để nhận dạng biển báo giao thông.
5 trang: Trang Chủ, Nhận Dạng Ảnh, Detection Pipeline, GradCAM, Thống Kê.

Chạy:
    streamlit run app/app.py
"""

import streamlit as st
import sys
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import io

# Fix Unicode encoding for Windows console (cp1252 can't handle emoji)
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", errors="replace", buffering=1)
if sys.stderr.encoding != "utf-8":
    sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", errors="replace", buffering=1)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---- Page Config ----
st.set_page_config(
    page_title="Nhận Dạng Biển Báo Giao Thông",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Load CSS ----
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ---- Cached Model Loading ----
@st.cache_resource
def load_detector(dataset="gtsrb"):
    """Load detector model (cached)."""
    if dataset == "vn":
        model_path = PROJECT_ROOT / "models" / "best_model_vn.pth"
    else:
        model_path = PROJECT_ROOT / "models" / "best_model.pth"
    if not model_path.exists():
        return None

    # Detect num_classes for VN model
    num_classes = 43
    if dataset == "vn":
        import torch
        checkpoint = torch.load(str(model_path), map_location="cpu")
        num_classes = checkpoint.get("num_classes", 43)

    from src.detection.detector import TrafficSignDetector
    detector = TrafficSignDetector(
        model_path=str(model_path),
        model_type="resnet18",
        confidence_threshold=0.3,
    )

    # Override class names for VN
    if dataset == "vn":
        vn_names = load_vn_class_names()
        if vn_names:
            detector.class_names = vn_names

    return detector


@st.cache_data
def load_vn_class_names():
    """Load VN class names mapping."""
    # Try vn_class_names.json
    path = PROJECT_ROOT / "data" / "vn_processed" / "vn_class_names.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {int(k): v for k, v in data.items()}

    # Try from training results
    path2 = PROJECT_ROOT / "reports" / "figures" / "vn_training_results.json"
    if path2.exists():
        with open(path2, "r", encoding="utf-8") as f:
            data = json.load(f)
            mapping = data.get("class_names_mapping", {})
            return {int(v): k for k, v in mapping.items()}

    return None


@st.cache_data
def load_evaluation_metrics():
    """Load evaluation metrics (cached)."""
    path = PROJECT_ROOT / "reports" / "figures" / "evaluation_metrics.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_training_history():
    """Load training history (cached)."""
    path = PROJECT_ROOT / "reports" / "figures" / "training_history.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def get_active_dataset():
    """Get the currently selected dataset from session state."""
    return st.session_state.get("active_dataset", "gtsrb")


def has_vn_model():
    """Check if VN model exists."""
    return (PROJECT_ROOT / "models" / "best_model_vn.pth").exists()


def main():
    # ---- Sidebar ----
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding: 0.5rem 0 1rem;'>
                <span style='font-size: 3rem;'>🚦</span>
                <h2 style='margin: 0.3rem 0 0; color: #c4b5fd; font-size: 1.1rem;'>
                    Traffic Sign AI
                </h2>
                <p style='color: #64748b; font-size: 0.75rem; margin: 0;'>
                    Deep Learning + Computer Vision
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        page = st.radio(
            "📌 Điều Hướng",
            [
                "🏠 Trang Chủ",
                "📸 Nhận Dạng Ảnh",
                "🔍 Detection Pipeline",
                "🧠 GradCAM",
                "📊 Thống Kê & Đánh Giá",
            ],
            index=0,
        )

        st.markdown("---")

        # Dataset selector
        dataset_options = ["🇩🇪 GTSRB (German)"]
        if has_vn_model():
            dataset_options.append("🇻🇳 Biển Báo Việt Nam")

        if len(dataset_options) > 1:
            selected = st.selectbox(
                "🗂️ Chọn Dataset",
                dataset_options,
                key="dataset_selector",
            )
            st.session_state["active_dataset"] = "vn" if "Việt Nam" in selected else "gtsrb"
        else:
            st.session_state["active_dataset"] = "gtsrb"

        active_ds = get_active_dataset()

        # Model status
        if active_ds == "vn":
            model_path = PROJECT_ROOT / "models" / "best_model_vn.pth"
            model_label = "🇻🇳 VN Model"
        else:
            model_path = PROJECT_ROOT / "models" / "best_model.pth"
            model_label = "🇩🇪 GTSRB"

        metrics = load_evaluation_metrics() if active_ds == "gtsrb" else None

        if model_path.exists():
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div style='color: #22c55e; font-size: 0.85rem;'>✅ Model Sẵn Sàng</div>
                    <div class='value' style='font-size: 1.3rem;'>ResNet18</div>
                    <div class='label'>{model_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if metrics:
                acc = metrics["overall"]["top1_accuracy"]
                st.markdown(
                    f"""
                    <div class='metric-card' style='margin-top: 0.5rem;'>
                        <div class='value' style='font-size: 1.5rem;'>{acc:.2f}%</div>
                        <div class='label'>Độ Chính Xác</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning("⚠️ Model chưa được huấn luyện")

        st.markdown("---")

        ds_info = "GTSRB · 43 Loại" if active_ds == "gtsrb" else "VNTS · Biển Báo VN"
        st.markdown(
            f"""
            <div style='text-align:center; padding-top: 0.5rem;'>
                <p style='color: #475569; font-size: 0.7rem; margin: 0;'>
                    📋 {ds_info}
                </p>
                <p style='color: #374151; font-size: 0.65rem; margin: 0.2rem 0 0;'>
                    © 2026 IPR Project Team
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Route Pages ----
    if page == "🏠 Trang Chủ":
        show_home_page()
    elif page == "📸 Nhận Dạng Ảnh":
        show_upload_page()
    elif page == "🔍 Detection Pipeline":
        show_detection_page()
    elif page == "🧠 GradCAM":
        show_gradcam_page()
    elif page == "📊 Thống Kê & Đánh Giá":
        show_statistics_page()


# ==============================================================
# PAGE 1: TRANG CHỦ
# ==============================================================
def show_home_page():
    # ---- Hero Section ----
    st.markdown(
        """
        <div class='hero-section'>
            <div class='hero-icon'>🚦</div>
            <div class='hero-title'>Hệ Thống Nhận Dạng Biển Báo Giao Thông</div>
            <p class='hero-subtitle'>
                Ứng dụng Deep Learning và Computer Vision để nhận dạng biển báo giao thông
                với độ chính xác cao, phục vụ mục tiêu an toàn giao thông.
            </p>
            <div style='margin-top: 1.5rem; display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;'>
                <span class='stat-highlight'>
                    <span class='stat-value'>43</span>
                    <span class='stat-label'>Loại Biển Báo</span>
                </span>
                <span class='stat-highlight'>
                    <span class='stat-value'>ResNet18</span>
                    <span class='stat-label'>Kiến Trúc</span>
                </span>
                <span class='stat-highlight'>
                    <span class='stat-value'>Real-time</span>
                    <span class='stat-label'>Phát Hiện</span>
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Feature Cards ----
    st.markdown("## ✨ Tính Năng Chính")
    col1, col2, col3, col4 = st.columns(4)

    features = [
        ("📸", "Nhận Dạng Ảnh", "Tải ảnh biển báo để nhận dạng với độ chính xác cao"),
        ("🔍", "Detection Pipeline", "Phát hiện &amp; nhận dạng nhiều biển báo trong 1 ảnh"),
        ("🧠", "GradCAM Visualization", "Hiển thị vùng model tập trung chú ý khi nhận dạng"),
        ("📊", "Thống Kê &amp; Đánh Giá", "Xem biểu đồ, confusion matrix và các chỉ số chi tiết"),
    ]

    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(
                f"""
                <div class='feature-card'>
                    <div class='icon'>{icon}</div>
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Pipeline Visualization ----
    st.markdown("## 🔄 Quy Trình Xử Lý")
    steps = [
        ("1", "Ảnh Đầu Vào", "🖼️"),
        ("2", "Phân Đoạn Màu", "🎨"),
        ("3", "Hình Thái Học", "⚙️"),
        ("4", "Phát Hiện Viền", "📐"),
        ("5", "Phân Loại CNN", "🧠"),
        ("6", "Kết Quả", "✅"),
    ]
    step_cols = st.columns(len(steps))
    for col, (num, title, icon) in zip(step_cols, steps):
        with col:
            st.markdown(
                f"""
                <div class='pipeline-step'>
                    <div>{icon}</div>
                    <div class='step-num'>{num}</div>
                    <div class='step-title'>{title}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Model Performance Summary ----
    metrics = load_evaluation_metrics()
    if metrics:
        st.markdown("## 🏆 Hiệu Suất Model")

        m1, m2, m3, m4, m5 = st.columns(5)
        overall = metrics["overall"]

        metric_data = [
            (m1, f"{overall['top1_accuracy']:.2f}%", "Top-1 Accuracy"),
            (m2, f"{overall['top5_accuracy']:.2f}%", "Top-5 Accuracy"),
            (m3, f"{overall['f1_macro']:.4f}", "F1 Macro"),
            (m4, f"{metrics['error_analysis']['total_errors']}", "Tổng Lỗi"),
            (m5, f"{overall['total_samples']}", "Mẫu Kiểm Tra"),
        ]

        for col, (value, label) in [(c, (v, l)) for c, v, l in metric_data]:
            with col:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='value'>{value}</div>
                        <div class='label'>{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Giới thiệu ----
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("## 📖 Giới Thiệu Dự Án")
        st.markdown(
            """
            Dự án sử dụng **Deep Learning** để nhận dạng biển báo giao thông,
            phục vụ cho mục tiêu **an toàn giao thông**. Hệ thống được xây dựng
            trên ngôn ngữ **Python** với các kỹ thuật:

            - 🖼️ **Xử lý ảnh**: CLAHE, Color Segmentation, Contour Detection
            - 🧠 **Mạng Neural**: CNN tùy chỉnh + Transfer Learning (ResNet18)
            - 📊 **Dataset**: GTSRB - 43 loại biển báo, ~39,000 ảnh
            - 🌐 **Web App**: Streamlit - giao diện trực quan, dễ sử dụng
            - 🎥 **Real-time**: Nhận dạng từ webcam trong thời gian thực
            """
        )

        st.markdown(
            """
            <div class='info-box'>
                <div class='info-title'>💡 Điểm Nổi Bật</div>
                <div class='info-text'>
                    Hệ thống hỗ trợ cả 2 dataset: GTSRB (43 loại biển báo Đức) và
                    Biển báo Việt Nam. Chuyển đổi dễ dàng qua sidebar.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("## 🔬 Kỹ Thuật IPR")

        techniques = [
            ("CLAHE", "Cân bằng histogram thích ứng"),
            ("Color Segmentation", "Phân đoạn theo màu HSV"),
            ("Contour Detection", "Phát hiện đường viền"),
            ("CNN", "Mạng nơ-ron tích chập"),
            ("Transfer Learning", "Fine-tune ResNet18"),
            ("Data Augmentation", "Tăng cường dữ liệu"),
            ("GradCAM", "Trực quan hóa attention"),
            ("Morphological Ops", "Lọc nhiễu ảnh"),
        ]

        for name, desc in techniques:
            st.markdown(
                f"""
                <div class='tech-badge'>
                    <b>{name}</b><br>
                    <span>{desc}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ---- Team ----
    st.markdown("## 👥 Nhóm Thực Hiện")
    members = [
        ("TV1", "Data Engineer", "Thu thập &amp; xử lý dữ liệu", "📊"),
        ("TV2", "ML Engineer", "Thiết kế &amp; huấn luyện model", "🧠"),
        ("TV3", "CV Engineer", "Detection &amp; Real-time", "👁️"),
        ("TV4", "Full-stack Dev", "Web App &amp; Tích hợp", "🌐"),
    ]

    cols = st.columns(4)
    for i, (name, role, task, emoji) in enumerate(members):
        with cols[i]:
            st.markdown(
                f"""
                <div class='member-card'>
                    <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>{emoji}</div>
                    <h3 style='color: #a78bfa; margin: 0.5rem 0 0.2rem; font-size: 1.15rem; font-weight: 700;'>{name}</h3>
                    <p style='color: #22d3ee; font-size: 0.85rem; margin: 0; font-weight: 600;'>{role}</p>
                    <p style='color: #64748b; font-size: 0.75rem; margin: 0.4rem 0 0;'>{task}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---- Footer ----
    st.markdown(
        """
        <div class='footer'>
            <p>🚦 Hệ Thống Nhận Dạng Biển Báo Giao Thông · © 2026 IPR Project Team</p>
            <p style='margin-top: 0.3rem;'>Xây dựng với Streamlit · PyTorch · OpenCV</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================
# PAGE 2: NHẬN DẠNG ẢNH
# ==============================================================
def show_upload_page():
    st.markdown(
        """
        <div class='page-header'>
            <h1>📸 Nhận Dạng Biển Báo Giao Thông</h1>
            <p class='subtitle'>Tải ảnh biển báo để hệ thống nhận dạng</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Chọn ảnh biển báo:",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=False,
        key="single_upload",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 🖼️ Ảnh đầu vào")
            st.image(image, use_column_width=True)

        with col2:
            st.markdown("### 🔍 Kết quả nhận dạng")

            detector = load_detector(get_active_dataset())
            if detector:
                with st.spinner("🔄 Đang nhận dạng..."):
                    try:
                        result = detector.classify_single_image(image)

                        # Main result
                        conf = result["confidence"]
                        conf_color = "#22c55e" if conf >= 0.8 else "#f59e0b" if conf >= 0.5 else "#ef4444"

                        st.markdown(
                            f"""
                            <div class='result-card'>
                                <p style='color: #64748b; font-size: 0.8rem; margin: 0;'>KẾT QUẢ</p>
                                <h2 style='color: #e2e8f0; margin: 0.3rem 0;'>{result['class_name']}</h2>
                                <p style='margin: 0;'>
                                    <span style='color: {conf_color}; font-size: 1.8rem; font-weight: 800;'>
                                        {conf:.1%}
                                    </span>
                                    <span style='color: #64748b; font-size: 0.85rem;'> độ tin cậy</span>
                                </p>
                                <p style='color: #64748b; font-size: 0.8rem; margin: 0.3rem 0 0;'>
                                    Class ID: {result['class_id']}
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown("<br>", unsafe_allow_html=True)

                        # Top 5
                        st.markdown("#### Top 5 dự đoán:")
                        for i, pred in enumerate(result["top5"]):
                            col_name, col_bar = st.columns([2, 3])
                            with col_name:
                                marker = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"  {i+1}."
                                st.markdown(f"**{marker} {pred['class_name']}**")
                            with col_bar:
                                st.progress(pred["confidence"])
                                st.caption(f"{pred['confidence']:.2%}")

                    except Exception as e:
                        st.error(f"Lỗi: {e}")
            else:
                st.warning(
                    "⚠️ Chưa có model! Chạy lệnh:\n"
                    "```\npython -m src.models.train\n```"
                )
    else:
        st.info("👆 Tải ảnh biển báo giao thông để bắt đầu nhận dạng.\n\n**Định dạng hỗ trợ:** JPG, JPEG, PNG, BMP")

        # List all classes
        with st.expander("📋 Danh sách 43 loại biển báo", expanded=False):
            from src.data.download_dataset import CLASS_NAMES
            cols = st.columns(3)
            for i, (class_id, name) in enumerate(CLASS_NAMES.items()):
                with cols[i % 3]:
                    st.markdown(f"`{class_id:02d}` {name}")


# ==============================================================
# PAGE 3: DETECTION PIPELINE
# ==============================================================
def show_detection_page():
    st.markdown(
        """
        <div class='page-header'>
            <h1>🔍 Detection Pipeline</h1>
            <p class='subtitle'>Phát hiện và nhận dạng nhiều biển báo trong 1 ảnh</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown(
        """
        **Pipeline:** Input → Color Segmentation (HSV) → Morphological Ops
        → Contour Detection → Bounding Box → Classification
        """
    )

    uploaded_file = st.file_uploader(
        "Tải ảnh có chứa biển báo giao thông:",
        type=["jpg", "jpeg", "png", "bmp"],
        key="detection_upload",
    )

    # Settings
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        confidence_threshold = st.slider("Ngưỡng Confidence", 0.1, 0.99, 0.5, 0.05)
    with col_s2:
        min_area = st.slider("Min Area (px)", 100, 2000, 500, 100)
    with col_s3:
        max_area = st.slider("Max Area (px)", 5000, 100000, 50000, 5000)

    if uploaded_file is not None:
        import cv2

        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        detector = load_detector(get_active_dataset())
        if detector:
            detector.confidence_threshold = confidence_threshold
            detector.min_area = min_area
            detector.max_area = max_area

            with st.spinner("🔄 Đang phát hiện biển báo..."):
                # Step 1: Detect regions
                bboxes = detector.detect_regions(img_bgr)

                # Step 2: Full detection
                detections = detector.detect(img_bgr)

                # Step 3: Draw
                annotated = detector.draw_detections(img_bgr, detections)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # Results
            tab1, tab2, tab3 = st.tabs(["📸 Kết Quả", "🔧 Quá Trình", "📋 Chi Tiết"])

            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Ảnh gốc")
                    st.image(image, use_column_width=True)
                with col2:
                    st.markdown(f"#### Kết quả ({len(detections)} biển báo)")
                    st.image(annotated_rgb, use_column_width=True)

            with tab2:
                st.markdown("#### Các bước xử lý")

                # Show HSV masks
                hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

                mask_cols = st.columns(4)
                color_configs = {
                    "Red (1)": (np.array([0, 70, 50]), np.array([10, 255, 255])),
                    "Red (2)": (np.array([170, 70, 50]), np.array([180, 255, 255])),
                    "Blue": (np.array([100, 70, 50]), np.array([130, 255, 255])),
                    "Yellow": (np.array([15, 70, 50]), np.array([35, 255, 255])),
                }

                for col, (name, (lower, upper)) in zip(mask_cols, color_configs.items()):
                    with col:
                        mask = cv2.inRange(hsv, lower, upper)
                        st.image(mask, caption=f"Mask: {name}", use_column_width=True)

                # Combined mask
                combined = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
                for lower, upper in color_configs.values():
                    combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lower, upper))

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
                combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

                c1, c2 = st.columns(2)
                with c1:
                    st.image(combined, caption="Combined Mask (after Morphology)", use_column_width=True)
                with c2:
                    # Draw bounding boxes on original
                    bbox_img = img_array.copy()
                    for (x, y, w, h) in bboxes:
                        cv2.rectangle(bbox_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    st.image(bbox_img, caption=f"Vùng phát hiện ({len(bboxes)} vùng)", use_column_width=True)

            with tab3:
                if detections:
                    for i, det in enumerate(detections):
                        x, y, w, h = det["bbox"]
                        crop = img_array[y:y+h, x:x+w]

                        st.markdown(f"---")
                        dc1, dc2 = st.columns([1, 3])
                        with dc1:
                            st.image(crop, caption=f"Biển báo #{i+1}", width=120)
                        with dc2:
                            st.markdown(f"**{det['class_name']}**")
                            st.markdown(f"Confidence: **{det['confidence']:.1%}**")
                            st.markdown(f"Class ID: `{det['class_id']}` | BBox: `({x}, {y}, {w}, {h})`")

                            if "top5" in det:
                                with st.expander("Top 5 dự đoán"):
                                    for pred in det["top5"]:
                                        st.markdown(f"- {pred['class_name']}: {pred['confidence']:.2%}")
                else:
                    st.info("Không phát hiện biển báo nào với ngưỡng hiện tại. Thử giảm ngưỡng confidence.")
        else:
            st.warning("⚠️ Chưa có model! Chạy `python -m src.models.train` trước.")


# ==============================================================
# PAGE 4: GRADCAM
# ==============================================================
def show_gradcam_page():
    st.markdown(
        """
        <div class='page-header'>
            <h1>🧠 GradCAM Visualization</h1>
            <p class='subtitle'>Xem vùng ảnh mà model tập trung chú ý khi nhận dạng</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown(
        """
        **GradCAM** (Gradient-weighted Class Activation Mapping) giúp hiển thị
        vùng mà model **chú ý** khi đưa ra dự đoán. Vùng **đỏ/vàng** là vùng
        quan trọng nhất, vùng **xanh** ít ảnh hưởng.
        """
    )

    uploaded_file = st.file_uploader(
        "Tải ảnh biển báo:",
        type=["jpg", "jpeg", "png", "bmp"],
        key="gradcam_upload",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        model_path = PROJECT_ROOT / "models" / "best_model.pth"

        if model_path.exists():
            with st.spinner("🔄 Đang tạo GradCAM..."):
                try:
                    import torch
                    import torchvision.transforms as transforms
                    import cv2
                    from src.utils.helpers import load_trained_model
                    from src.utils.visualization import GradCAM
                    from src.data.download_dataset import CLASS_NAMES

                    # Load model
                    device = torch.device("cpu")
                    model = load_trained_model(str(model_path), "resnet18", device=device)

                    # Transform
                    transform = transforms.Compose([
                        transforms.Resize((64, 64)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.3403, 0.3121, 0.3214],
                            std=[0.2724, 0.2608, 0.2669],
                        ),
                    ])

                    input_tensor = transform(image).unsqueeze(0)

                    # Get prediction
                    import torch.nn.functional as F
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = F.softmax(output, dim=1)
                        pred_class = output.argmax(dim=1).item()
                        confidence = probs[0, pred_class].item()

                    # Generate GradCAM
                    gradcam = GradCAM(model)
                    cam = gradcam.generate(input_tensor, class_idx=pred_class)

                    # Create heatmap
                    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                    # Resize original
                    img_resized = cv2.resize(img_array, (cam.shape[1], cam.shape[0]))
                    overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)

                    # Display result
                    st.markdown(
                        f"""
                        <div class='result-card'>
                            <h3 style='color: #e2e8f0; margin: 0;'>
                                Dự đoán: {CLASS_NAMES.get(pred_class, f'Class {pred_class}')}
                            </h3>
                            <p style='color: #06b6d4; font-size: 1.3rem; font-weight: 700; margin: 0.3rem 0 0;'>
                                {confidence:.1%} độ tin cậy
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("<br>", unsafe_allow_html=True)

                    # 3 columns
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.image(img_resized, caption="Ảnh gốc", use_column_width=True)
                    with c2:
                        st.image(heatmap, caption="GradCAM Heatmap", use_column_width=True)
                    with c3:
                        st.image(overlay, caption="Overlay", use_column_width=True)

                    st.markdown("---")
                    st.markdown("#### 📊 Giải thích")
                    st.markdown(
                        """
                        - 🔴 **Vùng đỏ/vàng**: Model tập trung chú ý nhiều nhất -> quyết định dự đoán
                        - 🔵 **Vùng xanh**: Ít ảnh hưởng đến dự đoán
                        - Model dùng **ResNet18 layer4** (layer conv cuối cùng) để tạo GradCAM
                        """
                    )

                    # Top 5 predictions
                    st.markdown("#### Top 5 Dự Đoán")
                    top5_probs, top5_indices = torch.topk(probs, 5)
                    for idx, prob in zip(top5_indices[0], top5_probs[0]):
                        name = CLASS_NAMES.get(idx.item(), f"Class {idx.item()}")
                        st.markdown(f"- `[{idx.item():02d}]` **{name}**: {prob.item():.2%}")

                except Exception as e:
                    st.error(f"Lỗi khi tạo GradCAM: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Chưa có model! Chạy `python -m src.models.train` trước.")
    else:
        st.info("👆 Tải ảnh biển báo để xem GradCAM visualization.")


# ==============================================================
# PAGE 5: THỐNG KÊ & ĐÁNH GIÁ
# ==============================================================
def show_statistics_page():
    st.markdown(
        """
        <div class='page-header'>
            <h1>📊 Thống Kê & Đánh Giá Model</h1>
            <p class='subtitle'>Kết quả đánh giá toàn diện model nhận dạng biển báo</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    metrics = load_evaluation_metrics()
    history = load_training_history()
    figures_dir = PROJECT_ROOT / "reports" / "figures"

    if metrics:
        overall = metrics["overall"]
        error = metrics["error_analysis"]

        # ---- Overall Metrics ----
        st.markdown("## 🎯 Chỉ Số Tổng Quan")

        cols = st.columns(6)
        metric_items = [
            (f"{overall['top1_accuracy']:.2f}%", "Top-1 Accuracy"),
            (f"{overall['top5_accuracy']:.2f}%", "Top-5 Accuracy"),
            (f"{overall['precision_macro']:.4f}", "Precision"),
            (f"{overall['recall_macro']:.4f}", "Recall"),
            (f"{overall['f1_macro']:.4f}", "F1 Macro"),
            (f"{error['total_errors']}/{overall['total_samples']}", "Lỗi"),
        ]

        for col, (value, label) in zip(cols, metric_items):
            with col:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='value'>{value}</div>
                        <div class='label'>{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Charts in Tabs ----
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Huấn Luyện",
            "🔢 Confusion Matrix",
            "📊 Theo Lớp",
            "📉 Confidence",
            "🔄 Cặp Hay Nhầm",
        ])

        with tab1:
            img_path = figures_dir / "training_curves.png"
            if img_path.exists():
                st.image(str(img_path), use_column_width=True)

                if history:
                    st.markdown("#### Tóm Tắt Huấn Luyện")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Epochs", len(history["train_loss"]))
                    with c2:
                        st.metric("Best Train Acc", f"{max(history['train_acc']):.2f}%")
                    with c3:
                        st.metric("Best Val Acc", f"{max(history['val_acc']):.2f}%")
                    with c4:
                        st.metric("Final LR", f"{history['lr'][-1]:.2e}")
            else:
                st.info("Chưa có dữ liệu training curves.")

        with tab2:
            img_path = figures_dir / "confusion_matrix.png"
            if img_path.exists():
                st.image(str(img_path), use_column_width=True)
            else:
                st.info("Chưa có confusion matrix. Chạy evaluation trước.")

        with tab3:
            img_path = figures_dir / "per_class_accuracy.png"
            if img_path.exists():
                st.image(str(img_path), use_column_width=True)

            # Per-class detail table
            if "per_class" in metrics:
                with st.expander("📋 Chi tiết từng class"):
                    per_class = metrics["per_class"]
                    import pandas as pd
                    df = pd.DataFrame([
                        {
                            "Class ID": int(k),
                            "Tên Biển Báo": v["name"],
                            "Accuracy (%)": v["accuracy"],
                            "Số Mẫu": v["support"],
                        }
                        for k, v in per_class.items()
                    ])
                    df = df.sort_values("Accuracy (%)")
                    st.dataframe(df, use_container_width=True, hide_index=True)

        with tab4:
            img_path = figures_dir / "confidence_distribution.png"
            if img_path.exists():
                st.image(str(img_path), use_column_width=True)

            st.markdown("#### Phân Tích Confidence")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Avg Confidence (Đúng)", f"{error['avg_correct_confidence']:.4f}")
            with c2:
                st.metric("Avg Confidence (Sai)", f"{error['avg_error_confidence']:.4f}")

        with tab5:
            img_path = figures_dir / "confused_pairs.png"
            if img_path.exists():
                st.image(str(img_path), use_column_width=True)

            if "confused_pairs" in metrics:
                st.markdown("#### Chi tiết các cặp hay nhầm")
                for pair in metrics["confused_pairs"]:
                    st.markdown(f"- **{pair['true']}** → **{pair['pred']}** ({pair['count']} lần)")

        st.markdown("---")

        # ---- Model Info ----
        st.markdown("## ⚙️ Cấu Hình Huấn Luyện")

        st.markdown(
            """
            | Tham số | Giá trị |
            |---------|---------|
            | Image Size | 64 x 64 |
            | Batch Size | 64 |
            | Learning Rate | 0.001 |
            | Optimizer | Adam (weight_decay=1e-4) |
            | LR Scheduler | StepLR (step=10, gamma=0.1) |
            | Early Stopping | patience=7, min_delta=0.001 |
            | Augmentation | Medium (rotation, affine, color jitter) |
            | Train/Val/Test | 70% / 15% / 15% |
            | Model | ResNet18 (pretrained ImageNet) |
            """
        )

    else:
        st.warning(
            "⚠️ Chưa có dữ liệu đánh giá!\n\n"
            "Chạy các lệnh sau:\n"
            "```\n"
            "set PYTHONIOENCODING=utf-8\n"
            "venv\\Scripts\\python.exe -m src.evaluation.evaluate\n"
            "```"
        )

    # ---- Dataset Info ----
    st.markdown("---")
    st.markdown("## 📁 Thông Tin Dataset")

    st.markdown(
        """
        | Thuộc tính | Giá trị |
        |-----------|---------|
        | Dataset | GTSRB (German Traffic Sign Recognition Benchmark) |
        | Số classes | 43 loại biển báo |
        | Tổng ảnh | ~39,209 (training) |
        | Image format | PPM (Portable Pixmap) |
        | Nguồn | INI Benchmark, Ruhr-University Bochum |
        """
    )


# ---- Entry Point ----
if __name__ == "__main__":
    main()
