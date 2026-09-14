"""
Helper Utilities
================
Các hàm hỗ trợ: load/save model, config, label mapping, v.v.

Sử dụng:
    from src.utils.helpers import load_config, get_device, load_trained_model
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path


def get_project_root():
    """Lấy đường dẫn root của project."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path=None):
    """
    Load file cấu hình YAML.

    Args:
        config_path: Đường dẫn file config (mặc định: config/config.yaml)

    Returns:
        Dict cấu hình
    """
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file không tồn tại: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_device():
    """
    Lấy device (cuda/cpu).

    Returns:
        torch.device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[GPU] Using: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("[CPU] Using CPU")
    return device


def load_trained_model(model_path, model_type="resnet18", num_classes=43, device=None):
    """
    Load model đã huấn luyện.

    Args:
        model_path: Đường dẫn file .pth
        model_type: Loại model
        num_classes: Số classes
        device: Device

    Returns:
        PyTorch model (eval mode)
    """
    from src.models.transfer_model import create_model

    if device is None:
        device = get_device()

    model = create_model(
        model_type=model_type,
        num_classes=num_classes,
        pretrained=False,
    )

    checkpoint = torch.load(str(model_path), map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "?")
        val_acc = checkpoint.get("val_acc", "?")
        print(f"[OK] Loaded model (epoch {epoch}, val_acc: {val_acc})")
    else:
        model.load_state_dict(checkpoint)
        print(f"[OK] Loaded model weights")

    model = model.to(device)
    model.eval()
    return model


def get_class_labels():
    """
    Lấy dict mapping class_id → tên biển báo.

    Returns:
        Dict {int: str}
    """
    from src.data.download_dataset import CLASS_NAMES
    return CLASS_NAMES


def count_parameters(model):
    """
    Đếm số tham số của model.

    Args:
        model: PyTorch model

    Returns:
        Dict thông tin tham số
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "total_readable": f"{total:,}",
        "trainable_readable": f"{trainable:,}",
    }


def set_seed(seed=42):
    """
    Set random seed cho reproducibility.

    Args:
        seed: Giá trị seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[SEED] Random seed set: {seed}")


def format_time(seconds):
    """
    Format thời gian từ giây sang dạng đọc được.

    Args:
        seconds: Số giây

    Returns:
        Chuỗi thời gian
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


def print_system_info():
    """In thông tin hệ thống."""
    print("=" * 50)
    print("[INFO] SYSTEM INFO")
    print("=" * 50)
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   NumPy: {np.__version__}")
    print(f"   Project Root: {get_project_root()}")
    print("=" * 50)


if __name__ == "__main__":
    print_system_info()
