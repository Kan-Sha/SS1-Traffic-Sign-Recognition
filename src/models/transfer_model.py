"""
Transfer Learning Model
=======================
Sử dụng pre-trained models (ResNet18/ResNet50) cho nhận dạng biển báo giao thông.
Fine-tune layers cuối cho 43 classes GTSRB.

Sử dụng:
    from src.models.transfer_model import TrafficSignTransfer
    model = TrafficSignTransfer(model_name="resnet18", num_classes=43)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class TrafficSignTransfer(nn.Module):
    """
    Transfer Learning model sử dụng ResNet pretrained.

    Args:
        model_name: Tên model ("resnet18", "resnet50")
        num_classes: Số lượng class (mặc định 43)
        pretrained: Sử dụng pretrained weights
        dropout: Tỷ lệ dropout
        freeze_backbone: Có freeze backbone hay không
    """

    def __init__(self, model_name="resnet18", num_classes=43,
                 pretrained=True, dropout=0.5, freeze_backbone=False):
        super(TrafficSignTransfer, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Load pretrained model
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features  # 512

        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features  # 2048

        else:
            raise ValueError(f"Model không hỗ trợ: {model_name}. Chọn 'resnet18' hoặc 'resnet50'.")

        # Freeze backbone nếu cần
        if freeze_backbone:
            self.freeze_backbone_layers()

        # Thay thế FC layer cuối
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
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
        return self.backbone(x)

    def predict(self, x):
        """
        Dự đoán với softmax probability.

        Args:
            x: Input tensor

        Returns:
            predictions, probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        return predictions, probs

    def freeze_backbone_layers(self):
        """Freeze tất cả layers của backbone (chỉ train FC layer)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🔒 Đã freeze backbone layers.")

    def unfreeze_backbone_layers(self, num_layers=None):
        """
        Unfreeze backbone layers để fine-tune.

        Args:
            num_layers: Số layers unfreeze từ cuối (None = unfreeze tất cả)
        """
        if num_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("🔓 Đã unfreeze tất cả backbone layers.")
        else:
            # Lấy danh sách children layers
            children = list(self.backbone.children())
            # Unfreeze từ cuối
            for child in children[-num_layers:]:
                for param in child.parameters():
                    param.requires_grad = True
            print(f"🔓 Đã unfreeze {num_layers} layers cuối.")

    def get_model_summary(self):
        """Trả về thông tin tóm tắt model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": f"Transfer-{self.model_name}",
            "backbone": self.model_name,
            "num_classes": self.num_classes,
            "total_params": f"{total_params:,}",
            "trainable_params": f"{trainable_params:,}",
            "frozen_params": f"{total_params - trainable_params:,}",
        }


def create_model(model_type="resnet18", num_classes=43, pretrained=True,
                 dropout=0.5, freeze_backbone=False):
    """
    Factory function tạo model dựa trên config.

    Args:
        model_type: "custom_cnn", "resnet18", "resnet50"
        num_classes: Số classes
        pretrained: Sử dụng pretrained weights
        dropout: Tỷ lệ dropout
        freeze_backbone: Freeze backbone layers

    Returns:
        PyTorch model
    """
    if model_type == "custom_cnn":
        from src.models.cnn_model import TrafficSignCNN
        return TrafficSignCNN(num_classes=num_classes, dropout=dropout)

    elif model_type in ("resnet18", "resnet50"):
        return TrafficSignTransfer(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )

    else:
        raise ValueError(f"Model type không hỗ trợ: {model_type}")


if __name__ == "__main__":
    # Test ResNet18
    print("=" * 50)
    print("🧠 Transfer Learning Model Test")
    print("=" * 50)

    model = TrafficSignTransfer(model_name="resnet18", num_classes=43, pretrained=True)
    print("\n📋 Model Summary:")
    for k, v in model.get_model_summary().items():
        print(f"   {k}: {v}")

    # Test forward pass
    dummy_input = torch.randn(4, 3, 64, 64)
    output = model(dummy_input)
    print(f"\n✅ Input shape: {dummy_input.shape}")
    print(f"✅ Output shape: {output.shape}")

    # Test factory
    print("\n🏭 Factory Test:")
    for m_type in ["custom_cnn", "resnet18", "resnet50"]:
        m = create_model(m_type, num_classes=43, pretrained=(m_type != "custom_cnn"))
        out = m(dummy_input)
        print(f"   {m_type}: output shape = {out.shape}")
