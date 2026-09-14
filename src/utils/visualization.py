"""
Visualization Utilities
=======================
Các hàm visualization cho training curves, confusion matrix, GradCAM, v.v.

Sử dụng:
    from src.utils.visualization import plot_training_curves, plot_confusion_matrix
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path


def plot_training_curves(history, save_path=None):
    """
    Vẽ biểu đồ training curves (Loss & Accuracy).

    Args:
        history: Dict {train_loss, val_loss, train_acc, val_acc}
        save_path: Đường dẫn lưu ảnh (tùy chọn)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Training Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2)
    ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-", label="Training Accuracy", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "r-", label="Validation Accuracy", linewidth=2)
    ax2.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"💾 Saved: {save_path}")

    plt.show()


def plot_confusion_matrix(cm, class_names=None, save_path=None, figsize=(15, 12)):
    """
    Vẽ confusion matrix.

    Args:
        cm: Confusion matrix (numpy array)
        class_names: Danh sách tên class
        save_path: Đường dẫn lưu ảnh
        figsize: Kích thước figure
    """
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    plt.figure(figsize=figsize)

    # Normalize
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={"shrink": 0.8},
    )

    plt.title("Confusion Matrix (Normalized)", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)

    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"💾 Saved: {save_path}")

    plt.show()


def plot_sample_predictions(images, true_labels, pred_labels, class_names=None,
                             n_samples=16, save_path=None):
    """
    Hiển thị grid ảnh với dự đoán.

    Args:
        images: numpy array ảnh (N, H, W, C)
        true_labels: Nhãn thực
        pred_labels: Nhãn dự đoán
        class_names: Dict class names
        n_samples: Số ảnh hiển thị
        save_path: Đường dẫn lưu ảnh
    """
    n = min(n_samples, len(images))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]

        if i < n:
            img = images[i]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)

            ax.imshow(img)

            true_label = true_labels[i]
            pred_label = pred_labels[i]
            correct = true_label == pred_label

            if class_names:
                true_name = class_names.get(int(true_label), str(true_label))
                pred_name = class_names.get(int(pred_label), str(pred_label))
            else:
                true_name = str(true_label)
                pred_name = str(pred_label)

            color = "green" if correct else "red"
            ax.set_title(f"T: {true_name}\nP: {pred_name}", fontsize=7, color=color)
        else:
            ax.axis("off")

        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Sample Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"💾 Saved: {save_path}")

    plt.show()


def plot_class_distribution(labels, class_names=None, save_path=None):
    """
    Vẽ biểu đồ phân phối classes.

    Args:
        labels: numpy array nhãn
        class_names: Dict class names
        save_path: Đường dẫn lưu ảnh
    """
    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(15, 6))

    if class_names:
        names = [class_names.get(int(u), str(u)) for u in unique]
    else:
        names = [str(u) for u in unique]

    bars = plt.bar(range(len(unique)), counts, color=plt.cm.viridis(np.linspace(0, 1, len(unique))))

    plt.title("Class Distribution", fontsize=16, fontweight="bold")
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(range(len(unique)), names, rotation=45, ha="right", fontsize=7)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"💾 Saved: {save_path}")

    plt.show()


class GradCAM:
    """
    Grad-CAM visualization - hiển thị vùng model chú ý trong ảnh.

    Args:
        model: PyTorch model
        target_layer: Layer target (mặc định: layer cuối của backbone)
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None

        # Xác định target layer
        if target_layer is None:
            # Mặc định: dùng layer conv cuối
            if hasattr(model, "backbone"):
                # Transfer Learning model
                target_layer = model.backbone.layer4[-1]
            elif hasattr(model, "conv_block3"):
                # Custom CNN
                target_layer = model.conv_block3
            else:
                raise ValueError("Không thể xác định target layer. Vui lòng chỉ định thủ công.")

        self.target_layer = target_layer

        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        Tạo GradCAM heatmap.

        Args:
            input_tensor: Input tensor (1, C, H, W)
            class_idx: Index class target (mặc định: class có probability cao nhất)

        Returns:
            Heatmap numpy array (H, W) với giá trị [0, 1]
        """
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # Global Average Pooling

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def visualize(self, input_tensor, original_image, class_idx=None, save_path=None):
        """
        Tạo và hiển thị GradCAM overlay.

        Args:
            input_tensor: Input tensor
            original_image: Ảnh gốc (numpy RGB)
            class_idx: Target class
            save_path: Đường dẫn lưu

        Returns:
            Ảnh overlay
        """
        cam = self.generate(input_tensor, class_idx)

        # Tạo heatmap
        heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Resize ảnh gốc
        img = original_image.copy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))

        # Overlay
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        # Plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        ax1.imshow(img)
        ax1.set_title("Original", fontsize=12)
        ax1.axis("off")

        ax2.imshow(cam, cmap="jet")
        ax2.set_title("GradCAM Heatmap", fontsize=12)
        ax2.axis("off")

        ax3.imshow(overlay)
        ax3.set_title("Overlay", fontsize=12)
        ax3.axis("off")

        plt.suptitle("GradCAM Visualization", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
            print(f"💾 Saved: {save_path}")

        plt.show()
        return overlay


if __name__ == "__main__":
    print("📊 Visualization Module")
    print("=" * 40)
    print("Các hàm có sẵn:")
    print("  - plot_training_curves(history)")
    print("  - plot_confusion_matrix(cm)")
    print("  - plot_sample_predictions(images, true, pred)")
    print("  - plot_class_distribution(labels)")
    print("  - GradCAM(model).visualize(input, image)")
