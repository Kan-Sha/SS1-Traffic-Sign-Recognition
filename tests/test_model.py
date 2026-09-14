"""
Test Model Module
=================
Unit tests cho CNN model và Transfer Learning model.
"""

import sys
import torch
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_model import TrafficSignCNN
from src.models.transfer_model import TrafficSignTransfer, create_model


class TestTrafficSignCNN:
    """Test class cho Custom CNN."""

    def setup_method(self):
        """Setup."""
        self.model = TrafficSignCNN(num_classes=43, input_size=64)
        self.dummy_input = torch.randn(4, 3, 64, 64)

    def test_output_shape(self):
        """Test output shape."""
        output = self.model(self.dummy_input)
        assert output.shape == (4, 43), f"Expected (4, 43), got {output.shape}"

    def test_single_input(self):
        """Test với batch size = 1."""
        single_input = torch.randn(1, 3, 64, 64)
        output = self.model(single_input)
        assert output.shape == (1, 43)

    def test_predict(self):
        """Test predict method."""
        preds, probs = self.model.predict(self.dummy_input)
        assert preds.shape == (4,)
        assert probs.shape == (4, 43)
        # Probabilities phải tổng = 1
        prob_sums = probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(4), atol=1e-5)

    def test_model_summary(self):
        """Test model summary."""
        summary = self.model.get_model_summary()
        assert summary["model_name"] == "TrafficSignCNN"
        assert summary["num_classes"] == 43

    def test_gradient_flow(self):
        """Test gradient có flow qua model."""
        output = self.model(self.dummy_input)
        loss = output.sum()
        loss.backward()

        has_grad = False
        for param in self.model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "Gradient không flow qua model"


class TestTrafficSignTransfer:
    """Test class cho Transfer Learning model."""

    def setup_method(self):
        """Setup."""
        self.model = TrafficSignTransfer(
            model_name="resnet18", num_classes=43, pretrained=False
        )
        self.dummy_input = torch.randn(2, 3, 64, 64)

    def test_output_shape(self):
        """Test output shape."""
        output = self.model(self.dummy_input)
        assert output.shape == (2, 43)

    def test_freeze_unfreeze(self):
        """Test freeze/unfreeze backbone."""
        self.model.freeze_backbone_layers()

        # Kiểm tra backbone bị freeze
        fc_params = list(self.model.backbone.fc.parameters())
        backbone_frozen = all(
            not p.requires_grad
            for name, p in self.model.backbone.named_parameters()
            if "fc" not in name
        )
        # FC layer vẫn phải trainable
        fc_trainable = all(p.requires_grad for p in fc_params)

        # Unfreeze
        self.model.unfreeze_backbone_layers()
        all_trainable = all(
            p.requires_grad for p in self.model.parameters()
        )
        assert all_trainable

    def test_model_summary(self):
        """Test model summary."""
        summary = self.model.get_model_summary()
        assert "resnet18" in summary["backbone"]


class TestCreateModel:
    """Test factory function."""

    def test_create_custom_cnn(self):
        """Test tạo Custom CNN."""
        model = create_model("custom_cnn", num_classes=43)
        output = model(torch.randn(1, 3, 64, 64))
        assert output.shape == (1, 43)

    def test_create_resnet18(self):
        """Test tạo ResNet18."""
        model = create_model("resnet18", num_classes=43, pretrained=False)
        output = model(torch.randn(1, 3, 64, 64))
        assert output.shape == (1, 43)

    def test_invalid_model(self):
        """Test model không hỗ trợ."""
        with pytest.raises(ValueError):
            create_model("invalid_model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
