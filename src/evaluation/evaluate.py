"""
Model Evaluation Module
========================
Đánh giá toàn diện model nhận dạng biển báo giao thông.
Tính metrics, confusion matrix, per-class accuracy, error analysis.

Sử dụng:
    python -m src.evaluation.evaluate
    python -m src.evaluation.evaluate --model models/best_model.pth --model-type resnet18
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    top_k_accuracy_score,
)

# Thêm project root vào path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transfer_model import create_model
from src.data.preprocessing import TrafficSignPreprocessor
from src.data.augmentation import get_test_transforms
from src.data.download_dataset import CLASS_NAMES


class ModelEvaluator:
    """
    Đánh giá toàn diện model nhận dạng biển báo giao thông.

    Args:
        model_path: Đường dẫn file model (.pth)
        model_type: Loại model ("custom_cnn", "resnet18", "resnet50")
        num_classes: Số lượng class
        device: Device (cuda/cpu)
    """

    def __init__(self, model_path, model_type="resnet18", num_classes=43, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.class_names = CLASS_NAMES
        self.results = {}

        # Load model
        print(f"📦 Đang load model: {model_path}")
        self.model = create_model(
            model_type=model_type,
            num_classes=num_classes,
            pretrained=False,
        )

        checkpoint = torch.load(str(model_path), map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.train_info = {
                "epoch": checkpoint.get("epoch", "N/A"),
                "val_acc": checkpoint.get("val_acc", "N/A"),
            }
            self.history = checkpoint.get("history", None)
        else:
            self.model.load_state_dict(checkpoint)
            self.train_info = {}
            self.history = None

        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model loaded trên {self.device}")

        if self.train_info:
            print(f"   Epoch: {self.train_info['epoch']}, Val Acc: {self.train_info['val_acc']:.2f}%")

    def evaluate(self, test_loader):
        """
        Đánh giá model trên test set.

        Args:
            test_loader: DataLoader cho test set

        Returns:
            Dict chứa tất cả kết quả đánh giá
        """
        print("\n" + "=" * 70)
        print("📊 ĐÁNH GIÁ MODEL TRÊN TEST SET")
        print("=" * 70)

        all_preds = []
        all_labels = []
        all_probs = []
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating", ncols=100):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # ---- Tính toán metrics ----
        accuracy = accuracy_score(all_labels, all_preds) * 100
        precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        # Top-5 accuracy
        top5_acc = top_k_accuracy_score(all_labels, all_probs, k=5, labels=range(self.num_classes)) * 100

        # Per-class metrics
        report_dict = classification_report(
            all_labels, all_preds,
            labels=range(self.num_classes),
            target_names=[self.class_names.get(i, f"Class {i}") for i in range(self.num_classes)],
            output_dict=True,
            zero_division=0,
        )
        report_text = classification_report(
            all_labels, all_preds,
            labels=range(self.num_classes),
            target_names=[f"[{i:02d}] {self.class_names.get(i, f'Class {i}')}" for i in range(self.num_classes)],
            zero_division=0,
        )

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=range(self.num_classes))

        # Per-class accuracy
        per_class_acc = {}
        for i in range(self.num_classes):
            mask = all_labels == i
            if mask.sum() > 0:
                class_acc = (all_preds[mask] == i).sum() / mask.sum() * 100
            else:
                class_acc = 0.0
            per_class_acc[i] = {
                "name": self.class_names.get(i, f"Class {i}"),
                "accuracy": class_acc,
                "support": int(mask.sum()),
            }

        # Top confused pairs
        confused_pairs = self._find_confused_pairs(cm, top_k=10)

        # Error analysis
        error_analysis = self._analyze_errors(all_labels, all_preds, all_probs)

        # ---- Lưu kết quả ----
        self.results = {
            "accuracy": accuracy,
            "top5_accuracy": top5_acc,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "classification_report": report_dict,
            "classification_report_text": report_text,
            "confusion_matrix": cm,
            "per_class_accuracy": per_class_acc,
            "confused_pairs": confused_pairs,
            "error_analysis": error_analysis,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs,
            "total_samples": total,
        }

        # ---- In kết quả ----
        self._print_results()

        return self.results

    def _find_confused_pairs(self, cm, top_k=10):
        """Tìm các cặp class hay bị nhầm lẫn nhất."""
        confused = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm[i, j] > 0:
                    confused.append({
                        "true_class": i,
                        "true_name": self.class_names.get(i, f"Class {i}"),
                        "pred_class": j,
                        "pred_name": self.class_names.get(j, f"Class {j}"),
                        "count": int(cm[i, j]),
                    })

        confused.sort(key=lambda x: x["count"], reverse=True)
        return confused[:top_k]

    def _analyze_errors(self, labels, preds, probs):
        """Phân tích các lỗi dự đoán."""
        error_mask = labels != preds
        n_errors = error_mask.sum()
        error_rate = n_errors / len(labels) * 100

        # Confidence phân phối cho các dự đoán đúng vs sai
        correct_confs = probs[~error_mask].max(axis=1)
        error_confs = probs[error_mask].max(axis=1) if n_errors > 0 else np.array([])

        # Classes có error rate cao nhất
        class_errors = {}
        for i in range(self.num_classes):
            mask = labels == i
            if mask.sum() > 0:
                class_error_rate = (preds[mask] != i).sum() / mask.sum() * 100
                class_errors[i] = {
                    "name": self.class_names.get(i, f"Class {i}"),
                    "error_rate": class_error_rate,
                    "n_errors": int((preds[mask] != i).sum()),
                    "n_total": int(mask.sum()),
                }

        # Sort by error rate
        worst_classes = sorted(class_errors.items(), key=lambda x: x[1]["error_rate"], reverse=True)[:10]

        return {
            "total_errors": int(n_errors),
            "error_rate": error_rate,
            "avg_correct_confidence": float(correct_confs.mean()) if len(correct_confs) > 0 else 0,
            "avg_error_confidence": float(error_confs.mean()) if len(error_confs) > 0 else 0,
            "worst_classes": worst_classes,
        }

    def _print_results(self):
        """In kết quả đánh giá."""
        r = self.results

        print("\n" + "=" * 70)
        print("📊 KẾT QUẢ ĐÁNH GIÁ")
        print("=" * 70)

        # Overall metrics
        print(f"\n🎯 Overall Metrics:")
        print(f"   Top-1 Accuracy : {r['accuracy']:.2f}%")
        print(f"   Top-5 Accuracy : {r['top5_accuracy']:.2f}%")
        print(f"   Precision (Macro): {r['precision_macro']:.4f}")
        print(f"   Recall (Macro)   : {r['recall_macro']:.4f}")
        print(f"   F1-Score (Macro) : {r['f1_macro']:.4f}")
        print(f"   F1-Score (Weighted): {r['f1_weighted']:.4f}")
        print(f"   Total Samples    : {r['total_samples']}")

        # Error analysis
        ea = r["error_analysis"]
        print(f"\n❌ Error Analysis:")
        print(f"   Total Errors    : {ea['total_errors']}")
        print(f"   Error Rate      : {ea['error_rate']:.2f}%")
        print(f"   Avg Correct Conf: {ea['avg_correct_confidence']:.4f}")
        print(f"   Avg Error Conf  : {ea['avg_error_confidence']:.4f}")

        # Top 5 worst classes
        print(f"\n⚠️ Top 5 Classes Có Error Rate Cao Nhất:")
        for class_id, info in ea["worst_classes"][:5]:
            print(f"   [{class_id:02d}] {info['name']}: "
                  f"{info['error_rate']:.1f}% ({info['n_errors']}/{info['n_total']})")

        # Top confused pairs
        print(f"\n🔄 Top 5 Cặp Hay Nhầm Lẫn:")
        for pair in r["confused_pairs"][:5]:
            print(f"   [{pair['true_class']:02d}] {pair['true_name']} "
                  f"→ [{pair['pred_class']:02d}] {pair['pred_name']} "
                  f"({pair['count']} lần)")

        # Per-class accuracy summary
        accs = [v["accuracy"] for v in r["per_class_accuracy"].values()]
        print(f"\n📊 Per-class Accuracy:")
        print(f"   Min : {min(accs):.2f}%")
        print(f"   Max : {max(accs):.2f}%")
        print(f"   Mean: {np.mean(accs):.2f}%")
        print(f"   Std : {np.std(accs):.2f}%")

        print("\n" + "=" * 70)

    def save_results(self, output_dir="reports/figures"):
        """
        Lưu tất cả kết quả và biểu đồ.

        Args:
            output_dir: Thư mục lưu kết quả
        """
        if not self.results:
            print("⚠️ Chưa có kết quả. Chạy evaluate() trước.")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n💾 Đang lưu kết quả vào: {output_dir}")

        # 1. Confusion Matrix
        self._save_confusion_matrix(output_dir / "confusion_matrix.png")

        # 2. Per-class Accuracy
        self._save_per_class_accuracy(output_dir / "per_class_accuracy.png")

        # 3. Confidence Distribution
        self._save_confidence_distribution(output_dir / "confidence_distribution.png")

        # 4. Training Curves (nếu có history)
        if self.history:
            self._save_training_curves(output_dir / "training_curves.png")

        # 5. Top Confused Pairs
        self._save_confused_pairs(output_dir / "confused_pairs.png")

        # 6. Metrics JSON
        self._save_metrics_json(output_dir / "evaluation_metrics.json")

        # 7. Classification Report text
        report_path = output_dir / "classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(self.results["classification_report_text"])
        print(f"   ✅ classification_report.txt")

        print(f"\n✅ Đã lưu tất cả kết quả vào: {output_dir}")

    def _save_confusion_matrix(self, path):
        """Lưu confusion matrix."""
        cm = self.results["confusion_matrix"]

        # Normalized confusion matrix
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

        fig, axes = plt.subplots(1, 2, figsize=(28, 12))

        # Raw
        sns.heatmap(
            cm, annot=False, cmap="Blues", ax=axes[0],
            xticklabels=[f"{i}" for i in range(self.num_classes)],
            yticklabels=[f"{i}" for i in range(self.num_classes)],
        )
        axes[0].set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Predicted", fontsize=12)
        axes[0].set_ylabel("True", fontsize=12)
        axes[0].tick_params(labelsize=7)

        # Normalized
        sns.heatmap(
            cm_norm, annot=False, cmap="Blues", ax=axes[1], vmin=0, vmax=1,
            xticklabels=[f"{i}" for i in range(self.num_classes)],
            yticklabels=[f"{i}" for i in range(self.num_classes)],
        )
        axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Predicted", fontsize=12)
        axes[1].set_ylabel("True", fontsize=12)
        axes[1].tick_params(labelsize=7)

        plt.tight_layout()
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✅ confusion_matrix.png")

    def _save_per_class_accuracy(self, path):
        """Lưu biểu đồ per-class accuracy."""
        pca = self.results["per_class_accuracy"]

        classes = sorted(pca.keys())
        accuracies = [pca[c]["accuracy"] for c in classes]
        names = [f"[{c:02d}]" for c in classes]

        fig, ax = plt.subplots(figsize=(18, 6))

        colors = []
        for acc in accuracies:
            if acc >= 95:
                colors.append("#22c55e")   # green
            elif acc >= 85:
                colors.append("#3b82f6")   # blue
            elif acc >= 70:
                colors.append("#f59e0b")   # yellow
            else:
                colors.append("#ef4444")   # red

        bars = ax.bar(range(len(classes)), accuracies, color=colors, edgecolor="white", linewidth=0.5)

        ax.set_title("Per-class Accuracy (%)", fontsize=16, fontweight="bold")
        ax.set_xlabel("Class ID", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 105)
        ax.axhline(y=np.mean(accuracies), color="red", linestyle="--", alpha=0.7,
                    label=f"Mean: {np.mean(accuracies):.1f}%")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        # Thêm giá trị trên mỗi bar
        for bar, acc in zip(bars, accuracies):
            if acc < 90:  # Chỉ hiển thị cho accuracy thấp
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                        f'{acc:.0f}', ha='center', va='bottom', fontsize=6, fontweight='bold')

        plt.tight_layout()
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✅ per_class_accuracy.png")

    def _save_confidence_distribution(self, path):
        """Lưu biểu đồ phân phối confidence."""
        labels = self.results["labels"]
        preds = self.results["predictions"]
        probs = self.results["probabilities"]

        correct_mask = labels == preds
        correct_confs = probs[correct_mask].max(axis=1)
        error_confs = probs[~correct_mask].max(axis=1) if (~correct_mask).sum() > 0 else np.array([])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(correct_confs, bins=50, alpha=0.7, color="#22c55e", label="Correct", density=True)
        if len(error_confs) > 0:
            axes[0].hist(error_confs, bins=50, alpha=0.7, color="#ef4444", label="Incorrect", density=True)
        axes[0].set_title("Confidence Distribution", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Confidence")
        axes[0].set_ylabel("Density")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Box plot
        data_to_plot = [correct_confs]
        labels_bp = ["Correct"]
        if len(error_confs) > 0:
            data_to_plot.append(error_confs)
            labels_bp.append("Incorrect")

        bp = axes[1].boxplot(data_to_plot, labels=labels_bp, patch_artist=True)
        colors_bp = ["#22c55e", "#ef4444"]
        for patch, color in zip(bp["boxes"], colors_bp[:len(bp["boxes"])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[1].set_title("Confidence Box Plot", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Confidence")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✅ confidence_distribution.png")

    def _save_training_curves(self, path):
        """Lưu training curves từ history."""
        h = self.history
        if not h or "train_loss" not in h:
            return

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        epochs = range(1, len(h["train_loss"]) + 1)

        # Loss
        axes[0].plot(epochs, h["train_loss"], "b-", label="Train", linewidth=2)
        axes[0].plot(epochs, h["val_loss"], "r-", label="Validation", linewidth=2)
        axes[0].set_title("Loss", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(epochs, h["train_acc"], "b-", label="Train", linewidth=2)
        axes[1].plot(epochs, h["val_acc"], "r-", label="Validation", linewidth=2)
        axes[1].set_title("Accuracy (%)", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Learning Rate
        if "lr" in h and h["lr"]:
            axes[2].plot(epochs, h["lr"], "g-", linewidth=2)
            axes[2].set_title("Learning Rate", fontsize=14, fontweight="bold")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("LR")
            axes[2].set_yscale("log")
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, "No LR data", ha="center", va="center", fontsize=14)
            axes[2].set_title("Learning Rate", fontsize=14, fontweight="bold")

        plt.suptitle("Training History", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✅ training_curves.png")

    def _save_confused_pairs(self, path):
        """Lưu biểu đồ top confused pairs."""
        pairs = self.results["confused_pairs"][:10]
        if not pairs:
            return

        fig, ax = plt.subplots(figsize=(14, 6))

        labels = [f"[{p['true_class']:02d}]→[{p['pred_class']:02d}]" for p in pairs]
        counts = [p["count"] for p in pairs]
        full_labels = [f"{p['true_name']}\n→ {p['pred_name']}" for p in pairs]

        bars = ax.barh(range(len(pairs)), counts, color="#f59e0b", edgecolor="white")
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(full_labels, fontsize=8)
        ax.set_xlabel("Số lần nhầm lẫn", fontsize=12)
        ax.set_title("Top 10 Cặp Class Hay Nhầm Lẫn", fontsize=14, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

        # Thêm số trên bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2.,
                    str(count), ha="left", va="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✅ confused_pairs.png")

    def _save_metrics_json(self, path):
        """Lưu metrics ra file JSON."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "model_info": self.train_info,
            "overall": {
                "top1_accuracy": round(self.results["accuracy"], 4),
                "top5_accuracy": round(self.results["top5_accuracy"], 4),
                "precision_macro": round(self.results["precision_macro"], 4),
                "recall_macro": round(self.results["recall_macro"], 4),
                "f1_macro": round(self.results["f1_macro"], 4),
                "f1_weighted": round(self.results["f1_weighted"], 4),
                "total_samples": self.results["total_samples"],
            },
            "error_analysis": {
                "total_errors": self.results["error_analysis"]["total_errors"],
                "error_rate": round(self.results["error_analysis"]["error_rate"], 4),
                "avg_correct_confidence": round(self.results["error_analysis"]["avg_correct_confidence"], 4),
                "avg_error_confidence": round(self.results["error_analysis"]["avg_error_confidence"], 4),
            },
            "per_class": {
                str(k): {
                    "name": v["name"],
                    "accuracy": round(v["accuracy"], 2),
                    "support": v["support"],
                }
                for k, v in self.results["per_class_accuracy"].items()
            },
            "confused_pairs": [
                {
                    "true": f"[{p['true_class']}] {p['true_name']}",
                    "pred": f"[{p['pred_class']}] {p['pred_name']}",
                    "count": p["count"],
                }
                for p in self.results["confused_pairs"]
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"   ✅ evaluation_metrics.json")


def load_processed_data(processed_dir):
    """
    Load dữ liệu đã xử lý từ file .npy.

    Args:
        processed_dir: Thư mục chứa file .npy

    Returns:
        Dict data_splits hoặc None nếu không có
    """
    processed_dir = Path(processed_dir)
    required_files = [
        "train_images.npy", "train_labels.npy",
        "val_images.npy", "val_labels.npy",
        "test_images.npy", "test_labels.npy",
    ]

    if all((processed_dir / f).exists() for f in required_files):
        print(f"⚡ Tìm thấy dữ liệu đã xử lý tại: {processed_dir}")
        data_splits = {}
        for split in ["train", "val", "test"]:
            images = np.load(str(processed_dir / f"{split}_images.npy"))
            labels = np.load(str(processed_dir / f"{split}_labels.npy"))
            data_splits[split] = {"images": images, "labels": labels}
            print(f"   {split}: {len(images)} ảnh, {len(set(labels))} classes")
        return data_splits
    return None


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Đánh giá model nhận dạng biển báo giao thông")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="Đường dẫn model")
    parser.add_argument("--model-type", type=str, default="resnet18",
                        choices=["custom_cnn", "resnet18", "resnet50"],
                        help="Loại model")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Thư mục dữ liệu raw")
    parser.add_argument("--processed-dir", type=str, default="data/processed",
                        help="Thư mục dữ liệu đã xử lý (.npy)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--image-size", type=int, default=64,
                        help="Kích thước ảnh")
    parser.add_argument("--output-dir", type=str, default="reports/figures",
                        help="Thư mục lưu kết quả")
    args = parser.parse_args()

    print("🚦 Traffic Sign Recognition - Model Evaluation")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📍 Device: {device}")

    # Check model file
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Không tìm thấy model: {model_path}")
        print("   Vui lòng huấn luyện model trước: python -m src.models.train")
        return

    # Try loading processed data first (much faster)
    data_splits = load_processed_data(args.processed_dir)

    if data_splits is None:
        # Fallback: load & process from raw
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"❌ Không tìm thấy dữ liệu: {data_dir}")
            print("   Vui lòng tải dataset: python -m src.data.download_dataset")
            return

        print(f"\n📁 Đang tải và xử lý dữ liệu từ: {data_dir}")
        preprocessor = TrafficSignPreprocessor(image_size=args.image_size)
        data_splits = preprocessor.prepare_dataset(str(data_dir), args.processed_dir)

    # Create test loader
    preprocessor = TrafficSignPreprocessor(image_size=args.image_size)
    test_transform = get_test_transforms(args.image_size)
    dataloaders = preprocessor.create_dataloaders(
        data_splits,
        batch_size=args.batch_size,
        test_transform=test_transform,
    )

    print(f"\n   Test samples: {len(data_splits['test']['images'])}")
    print(f"   Test batches: {len(dataloaders['test'])}")

    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=str(model_path),
        model_type=args.model_type,
        device=device,
    )

    # Run evaluation
    results = evaluator.evaluate(dataloaders["test"])

    # Save results
    evaluator.save_results(args.output_dir)

    # Generate report
    print("\n📝 Đang tạo báo cáo...")
    from src.evaluation.report_generator import generate_report
    generate_report(
        metrics_path=str(Path(args.output_dir) / "evaluation_metrics.json"),
        output_path="docs/report.md",
        figures_dir=args.output_dir,
    )

    print(f"\n🎉 Evaluation hoàn tất!")


if __name__ == "__main__":
    main()
