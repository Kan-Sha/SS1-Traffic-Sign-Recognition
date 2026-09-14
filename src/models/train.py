"""
Training Pipeline
=================
Pipeline huấn luyện model nhận dạng biển báo giao thông.
Hỗ trợ early stopping, learning rate scheduling, model checkpointing.

Sử dụng:
    python -m src.models.train --model resnet18 --epochs 30
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Thêm project root vào path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transfer_model import create_model
from src.data.preprocessing import TrafficSignPreprocessor, get_default_transforms
from src.data.augmentation import get_train_transforms, get_test_transforms
from sklearn.metrics import classification_report, confusion_matrix


class EarlyStopping:
    """
    Early Stopping để ngăn overfitting.

    Args:
        patience: Số epoch chờ trước khi dừng
        min_delta: Thay đổi tối thiểu để coi là cải thiện
    """

    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n⚠️ Early stopping triggered sau {self.patience} epochs không cải thiện.")
        else:
            self.best_loss = val_loss
            self.counter = 0


class Trainer:
    """
    Trainer cho model nhận dạng biển báo.

    Args:
        model: PyTorch model
        device: Device (cuda/cpu)
        save_dir: Thư mục lưu model
    """

    def __init__(self, model, device=None, save_dir="models"):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Lịch sử training
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

    def train(self, train_loader, val_loader, epochs=30, lr=0.001,
              weight_decay=1e-4, patience=7, scheduler_type="step"):
        """
        Huấn luyện model.

        Args:
            train_loader: DataLoader cho training
            val_loader: DataLoader cho validation
            epochs: Số epoch
            lr: Learning rate
            weight_decay: Weight decay cho optimizer
            patience: Patience cho early stopping
            scheduler_type: Loại learning rate scheduler

        Returns:
            history: Dict lịch sử training
        """
        # Loss & Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Learning Rate Scheduler
        if scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3
            )
        else:
            scheduler = None

        # Early Stopping
        early_stopping = EarlyStopping(patience=patience)

        # Best model tracking
        best_val_acc = 0.0
        best_model_path = self.save_dir / "best_model.pth"

        print("=" * 70)
        print(f"🚀 BẮT ĐẦU HUẤN LUYỆN")
        print(f"   Device: {self.device}")
        print(f"   Model: {type(self.model).__name__}")
        print(f"   Epochs: {epochs}")
        print(f"   Learning Rate: {lr}")
        print(f"   Scheduler: {scheduler_type}")
        print(f"   Patience: {patience}")
        print("=" * 70)

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # ---- TRAINING ----
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, optimizer, epoch, epochs
            )

            # ---- VALIDATION ----
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)

            # Learning rate step
            current_lr = optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)

            if scheduler:
                if scheduler_type == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Lưu lịch sử
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # In kết quả epoch
            print(
                f"Epoch [{epoch}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.6f}"
            )

            # Lưu best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_checkpoint(best_model_path, epoch, val_acc, optimizer)
                print(f"   💾 Saved best model (Val Acc: {val_acc:.2f}%)")

            # Early Stopping check
            early_stopping(val_loss)
            if early_stopping.should_stop:
                break

        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"✅ HUẤN LUYỆN HOÀN TẤT")
        print(f"   Thời gian: {total_time / 60:.1f} phút")
        print(f"   Best Val Accuracy: {best_val_acc:.2f}%")
        print(f"   Model saved: {best_model_path}")
        print("=" * 70)

        return self.history

    def _train_epoch(self, train_loader, criterion, optimizer, epoch, total_epochs):
        """Huấn luyện 1 epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]",
                     leave=False, ncols=100)

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Thống kê
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.1f}%",
            )

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def _validate_epoch(self, val_loader, criterion):
        """Đánh giá trên validation set."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def evaluate(self, test_loader):
        """
        Đánh giá model trên test set.

        Args:
            test_loader: DataLoader cho test set

        Returns:
            Dict kết quả đánh giá
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        correct = 0
        total = 0

        print("\n📊 Đang đánh giá trên test set...")

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing", ncols=100):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100.0 * correct / total
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Classification report
        report = classification_report(all_labels, all_preds, output_dict=True)
        cm = confusion_matrix(all_labels, all_preds)

        print(f"\n✅ Test Accuracy: {accuracy:.2f}%")
        print(f"   Precision (macro): {report['macro avg']['precision']:.4f}")
        print(f"   Recall (macro): {report['macro avg']['recall']:.4f}")
        print(f"   F1-Score (macro): {report['macro avg']['f1-score']:.4f}")

        return {
            "accuracy": accuracy,
            "predictions": all_preds,
            "labels": all_labels,
            "classification_report": report,
            "confusion_matrix": cm,
        }

    def _save_checkpoint(self, path, epoch, val_acc, optimizer):
        """Lưu model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "history": self.history,
        }, str(path))

    def load_checkpoint(self, path):
        """
        Load model checkpoint.

        Args:
            path: Đường dẫn file checkpoint
        """
        checkpoint = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        print(f"✅ Loaded model: epoch {checkpoint['epoch']}, val_acc: {checkpoint['val_acc']:.2f}%")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Huấn luyện model nhận dạng biển báo giao thông")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["custom_cnn", "resnet18", "resnet50"],
                        help="Loại model (mặc định: resnet18)")
    parser.add_argument("--epochs", type=int, default=30, help="Số epoch")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--image-size", type=int, default=64, help="Kích thước ảnh")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Thư mục dữ liệu")
    parser.add_argument("--save-dir", type=str, default="models", help="Thư mục lưu model")
    parser.add_argument("--augment-level", type=str, default="medium",
                        choices=["light", "medium", "heavy"],
                        help="Mức độ augmentation")
    args = parser.parse_args()

    print("🚦 Traffic Sign Recognition - Training Pipeline")
    print("=" * 50)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📍 Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load & preprocess data
    print("\n📁 Đang tải dữ liệu...")
    preprocessor = TrafficSignPreprocessor(image_size=args.image_size)
    data_splits = preprocessor.prepare_dataset(args.data_dir, "data/processed")

    # 2. Create transforms
    train_transform = get_train_transforms(args.image_size, args.augment_level)
    test_transform = get_test_transforms(args.image_size)

    # 3. Create DataLoaders
    dataloaders = preprocessor.create_dataloaders(
        data_splits,
        batch_size=args.batch_size,
        train_transform=train_transform,
        test_transform=test_transform,
    )

    print(f"   Train batches: {len(dataloaders['train'])}")
    print(f"   Val batches: {len(dataloaders['val'])}")
    print(f"   Test batches: {len(dataloaders['test'])}")

    # 4. Create model
    print(f"\n🧠 Đang tạo model: {args.model}")
    model = create_model(
        model_type=args.model,
        num_classes=43,
        pretrained=True,
        dropout=0.5,
    )

    # 5. Train
    trainer = Trainer(model, device=device, save_dir=args.save_dir)
    history = trainer.train(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )

    # 6. Evaluate
    results = trainer.evaluate(dataloaders["test"])

    # 7. Save final results
    print("\n💾 Đang lưu kết quả...")
    results_dir = Path("reports/figures")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save history
    import json
    with open(results_dir / "training_history.json", "w") as f:
        json.dump({
            "train_loss": history["train_loss"],
            "train_acc": history["train_acc"],
            "val_loss": history["val_loss"],
            "val_acc": history["val_acc"],
            "lr": history["lr"],
        }, f, indent=2)

    print(f"\n🎉 Hoàn tất! Model đã lưu tại: {args.save_dir}/best_model.pth")


if __name__ == "__main__":
    main()
