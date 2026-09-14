"""
Training Pipeline cho dataset Biển Báo Việt Nam
=================================================
Fine-tune ResNet18 trên dataset VN đã crop.
Load data từ data/vn_processed/ (ImageFolder format).

Chạy:
    python -m src.models.train_vn --epochs 20
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transfer_model import create_model


class VNTrainer:
    """Trainer cho model biển báo Việt Nam."""

    def __init__(self, num_classes, model_type="resnet18", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.model_type = model_type

        # Create model
        self.model = create_model(
            model_type=model_type,
            num_classes=num_classes,
            pretrained=True,
            dropout=0.5,
        )
        self.model = self.model.to(self.device)

        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "lr": [],
        }

    def train(self, train_loader, val_loader, epochs=20, lr=0.001, patience=7):
        """Training loop."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        best_val_acc = 0
        patience_counter = 0

        for epoch in range(epochs):
            # ---- Train ----
            self.model.train()
            train_loss, train_correct, train_total = 0, 0, 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                train_correct += preds.eq(targets).sum().item()
                train_total += targets.size(0)

                pbar.set_postfix({
                    "loss": f"{train_loss/train_total:.4f}",
                    "acc": f"{100.*train_correct/train_total:.2f}%"
                })

            # ---- Validate ----
            self.model.eval()
            val_loss, val_correct, val_total = 0, 0, 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = outputs.max(1)
                    val_correct += preds.eq(targets).sum().item()
                    val_total += targets.size(0)

            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            avg_train_loss = train_loss / train_total
            avg_val_loss = val_loss / val_total
            current_lr = optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(round(avg_train_loss, 4))
            self.history["train_acc"].append(round(train_acc, 2))
            self.history["val_loss"].append(round(avg_val_loss, 4))
            self.history["val_acc"].append(round(val_acc, 2))
            self.history["lr"].append(current_lr)

            print(f"\n  Epoch {epoch+1}: "
                  f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                  f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%, "
                  f"LR={current_lr:.6f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self._save_checkpoint(epoch, val_acc)
                print(f"  🏆 Best model! Val Acc = {val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n  ⚠️ Early stopping sau {patience} epochs không cải thiện")
                    break

            scheduler.step()

        return self.history

    def _save_checkpoint(self, epoch, val_acc):
        """Lưu checkpoint."""
        save_dir = PROJECT_ROOT / "models"
        save_dir.mkdir(parents=True, exist_ok=True)

        path = save_dir / "best_model_vn.pth"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "val_acc": val_acc,
            "num_classes": self.num_classes,
            "model_type": self.model_type,
            "history": self.history,
            "dataset": "vnts",
        }, str(path))

    def evaluate(self, test_loader):
        """Evaluate trên test set."""
        self.model.eval()
        correct, total = 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Evaluating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, preds = outputs.max(1)

                correct += preds.eq(targets).sum().item()
                total += targets.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        accuracy = 100. * correct / total
        print(f"\n  📊 Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
        return accuracy, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(description="Train model biển báo Việt Nam")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["custom_cnn", "resnet18", "resnet50"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "vn_processed"

    print("=" * 60)
    print("🇻🇳 Vietnam Traffic Sign - Training Pipeline")
    print("=" * 60)
    print(f"  📍 Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"  📁 Data: {data_dir}")

    if not data_dir.exists():
        print(f"\n  ❌ Chưa có dữ liệu! Chạy convert trước:")
        print(f"     python -m src.data.convert_yolo_to_classification")
        sys.exit(1)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomHorizontalFlip(p=0.0),  # Không flip biển báo
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if not train_dir.exists():
        print(f"  ❌ Không tìm thấy {train_dir}")
        sys.exit(1)

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=test_transform)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=test_transform)

    num_classes = len(train_dataset.classes)

    print(f"\n  📊 Dataset:")
    print(f"     Train: {len(train_dataset)} ảnh")
    print(f"     Val:   {len(val_dataset)} ảnh")
    print(f"     Test:  {len(test_dataset)} ảnh")
    print(f"     Classes: {num_classes}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # Train
    print(f"\n  🧠 Model: {args.model} | Classes: {num_classes}")
    trainer = VNTrainer(
        num_classes=num_classes,
        model_type=args.model,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    history = trainer.train(
        train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
    )

    # Evaluate
    accuracy, preds, targets = trainer.evaluate(test_loader)

    # Save history + metadata
    results_dir = PROJECT_ROOT / "reports" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)

    vn_results = {
        "dataset": "vnts",
        "num_classes": num_classes,
        "class_names_mapping": {str(i): c for i, c in enumerate(train_dataset.classes)},
        "test_accuracy": accuracy,
        "history": history,
    }

    with open(results_dir / "vn_training_results.json", "w", encoding="utf-8") as f:
        json.dump(vn_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"🎉 HOÀN TẤT!")
    print(f"{'='*60}")
    print(f"  💾 Model: models/best_model_vn.pth")
    print(f"  📊 Test Accuracy: {accuracy:.2f}%")
    print(f"  📋 Results: {results_dir / 'vn_training_results.json'}")


if __name__ == "__main__":
    main()
