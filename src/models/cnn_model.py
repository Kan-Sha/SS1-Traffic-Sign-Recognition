"""
Custom CNN Model
================
Mô hình CNN tùy chỉnh cho nhận dạng biển báo giao thông.
Kiến trúc: 3 Conv Blocks + BatchNorm + Dropout + Fully Connected.

Sử dụng:
    from src.models.cnn_model import TrafficSignCNN
    model = TrafficSignCNN(num_classes=43)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Block tích chập: Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → MaxPool → Dropout.

    Args:
        in_channels: Số kênh đầu vào
        out_channels: Số kênh đầu ra
        dropout: Tỷ lệ dropout
    """

    def __init__(self, in_channels, out_channels, dropout=0.25):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        return x


class TrafficSignCNN(nn.Module):
    """
    Custom CNN cho nhận dạng biển báo giao thông.

    Architecture:
        - Input: (batch, 3, 64, 64)
        - Conv Block 1: 3 → 32 channels, output: (batch, 32, 32, 32)
        - Conv Block 2: 32 → 64 channels, output: (batch, 64, 16, 16)
        - Conv Block 3: 64 → 128 channels, output: (batch, 128, 8, 8)
        - Flatten: (batch, 128 * 8 * 8) = (batch, 8192)
        - FC1: 8192 → 512
        - FC2: 512 → 256
        - Output: 256 → num_classes

    Args:
        num_classes: Số lượng class (mặc định 43 cho GTSRB)
        dropout: Tỷ lệ dropout (mặc định 0.5)
        input_size: Kích thước ảnh đầu vào (mặc định 64)
    """

    def __init__(self, num_classes=43, dropout=0.5, input_size=64):
        super(TrafficSignCNN, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size

        # Conv Blocks
        self.conv_block1 = ConvBlock(3, 32, dropout=0.25)
        self.conv_block2 = ConvBlock(32, 64, dropout=0.25)
        self.conv_block3 = ConvBlock(64, 128, dropout=0.25)

        # Tính kích thước feature map sau 3 lần MaxPool
        feature_size = input_size // (2 ** 3)  # 64 → 8
        flatten_size = 128 * feature_size * feature_size  # 128 * 8 * 8

        # Fully Connected Layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch, 3, H, W)

        Returns:
            Output logits (batch, num_classes)
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x

    def predict(self, x):
        """
        Dự đoán với softmax probability.

        Args:
            x: Input tensor

        Returns:
            predictions: class indices
            probabilities: softmax probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        return predictions, probs

    def get_model_summary(self):
        """Trả về thông tin tóm tắt model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": "TrafficSignCNN",
            "num_classes": self.num_classes,
            "input_size": f"(3, {self.input_size}, {self.input_size})",
            "total_params": f"{total_params:,}",
            "trainable_params": f"{trainable_params:,}",
        }


if __name__ == "__main__":
    # Test model
    model = TrafficSignCNN(num_classes=43, input_size=64)
    print("📋 Model Summary:")
    for k, v in model.get_model_summary().items():
        print(f"   {k}: {v}")

    # Test forward pass
    dummy_input = torch.randn(4, 3, 64, 64)
    output = model(dummy_input)
    print(f"\n✅ Input shape: {dummy_input.shape}")
    print(f"✅ Output shape: {output.shape}")

    # Test predict
    preds, probs = model.predict(dummy_input)
    print(f"✅ Predictions: {preds}")
    print(f"✅ Top probability: {probs.max(dim=1).values}")
