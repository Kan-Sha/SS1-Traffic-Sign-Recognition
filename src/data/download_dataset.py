"""
Download GTSRB (German Traffic Sign Recognition Benchmark) Dataset
==================================================================
Script tải và giải nén dataset GTSRB cho việc huấn luyện model nhận dạng biển báo.

Sử dụng:
    python -m src.data.download_dataset
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm


# GTSRB Dataset URLs
DATASET_URLS = {
    "train": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip",
    "test": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip",
    "test_labels": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip",
}

# Tên 43 loại biển báo GTSRB
CLASS_NAMES = {
    0: "Giới hạn tốc độ (20km/h)",
    1: "Giới hạn tốc độ (30km/h)",
    2: "Giới hạn tốc độ (50km/h)",
    3: "Giới hạn tốc độ (60km/h)",
    4: "Giới hạn tốc độ (70km/h)",
    5: "Giới hạn tốc độ (80km/h)",
    6: "Hết giới hạn tốc độ (80km/h)",
    7: "Giới hạn tốc độ (100km/h)",
    8: "Giới hạn tốc độ (120km/h)",
    9: "Cấm vượt",
    10: "Cấm vượt (xe > 3.5 tấn)",
    11: "Ưu tiên tại ngã tư",
    12: "Đường ưu tiên",
    13: "Nhường đường",
    14: "Dừng lại (STOP)",
    15: "Cấm xe cộ",
    16: "Cấm xe > 3.5 tấn",
    17: "Cấm đi vào",
    18: "Nguy hiểm chung",
    19: "Đường cong nguy hiểm bên trái",
    20: "Đường cong nguy hiểm bên phải",
    21: "Đường cong kép",
    22: "Đường gồ ghề",
    23: "Đường trơn trượt",
    24: "Đường hẹp bên phải",
    25: "Công trường",
    26: "Đèn tín hiệu giao thông",
    27: "Người đi bộ",
    28: "Trẻ em qua đường",
    29: "Xe đạp qua đường",
    30: "Cẩn thận băng/tuyết",
    31: "Động vật hoang dã qua đường",
    32: "Hết tất cả giới hạn",
    33: "Bắt buộc rẽ phải",
    34: "Bắt buộc rẽ trái",
    35: "Bắt buộc đi thẳng",
    36: "Bắt buộc đi thẳng hoặc rẽ phải",
    37: "Bắt buộc đi thẳng hoặc rẽ trái",
    38: "Bắt buộc đi bên phải",
    39: "Bắt buộc đi bên trái",
    40: "Bắt buộc vòng xuyến",
    41: "Hết cấm vượt",
    42: "Hết cấm vượt (xe > 3.5 tấn)",
}


class DownloadProgressBar(tqdm):
    """Progress bar cho việc tải file."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def get_project_root():
    """Lấy đường dẫn gốc của project."""
    return Path(__file__).resolve().parent.parent.parent


def download_file(url, output_path):
    """
    Tải file từ URL với progress bar.

    Args:
        url: URL của file cần tải
        output_path: Đường dẫn lưu file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"  ✓ File đã tồn tại: {output_path.name}")
        return

    print(f"  ⬇ Đang tải: {output_path.name}")
    try:
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=output_path.name
        ) as t:
            urllib.request.urlretrieve(url, filename=str(output_path), reporthook=t.update_to)
        print(f"  ✓ Tải thành công: {output_path.name}")
    except Exception as e:
        print(f"  ✗ Lỗi tải {output_path.name}: {e}")
        if output_path.exists():
            output_path.unlink()
        raise


def extract_zip(zip_path, extract_to):
    """
    Giải nén file ZIP.

    Args:
        zip_path: Đường dẫn file ZIP
        extract_to: Thư mục giải nén
    """
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    print(f"  📦 Đang giải nén: {zip_path.name}")
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zip_ref:
            zip_ref.extractall(str(extract_to))
        print(f"  ✓ Giải nén thành công vào: {extract_to}")
    except Exception as e:
        print(f"  ✗ Lỗi giải nén: {e}")
        raise


def download_gtsrb(data_dir=None):
    """
    Tải và giải nén toàn bộ dataset GTSRB.

    Args:
        data_dir: Thư mục lưu dữ liệu (mặc định: data/raw)
    """
    if data_dir is None:
        data_dir = get_project_root() / "data" / "raw"

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🚦 GTSRB Dataset Downloader")
    print("=" * 60)
    print(f"📁 Thư mục lưu: {data_dir}")
    print(f"📊 Số loại biển báo: {len(CLASS_NAMES)}")
    print()

    for name, url in DATASET_URLS.items():
        filename = url.split("/")[-1]
        zip_path = data_dir / filename

        print(f"\n[{name.upper()}]")
        download_file(url, zip_path)
        extract_zip(zip_path, data_dir)

    print("\n" + "=" * 60)
    print("✅ Tải dataset hoàn tất!")
    print(f"📁 Dữ liệu được lưu tại: {data_dir}")
    print("=" * 60)

    # Thống kê
    count_images(data_dir)


def count_images(data_dir):
    """Đếm số lượng ảnh trong dataset."""
    data_dir = Path(data_dir)
    extensions = {".ppm", ".png", ".jpg", ".jpeg"}
    total = 0

    for ext in extensions:
        count = len(list(data_dir.rglob(f"*{ext}")))
        if count > 0:
            print(f"  📷 Ảnh {ext}: {count}")
        total += count

    print(f"  📊 Tổng số ảnh: {total}")


def get_class_name(class_id):
    """
    Lấy tên biển báo từ class ID.

    Args:
        class_id: ID của class (0-42)

    Returns:
        Tên biển báo bằng tiếng Việt
    """
    return CLASS_NAMES.get(class_id, f"Không xác định ({class_id})")


if __name__ == "__main__":
    download_gtsrb()
