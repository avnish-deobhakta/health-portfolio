"""
Model definitions for diabetic retinopathy screening experiments.

Includes standard timm models and RETFound wrappers with layer-wise
learning rate decay.
"""

import torch
import torch.nn as nn
import timm


class RETFoundBinary(nn.Module):
    """RETFound with binary classification head."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.classifier = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        features = self.base(x).last_hidden_state[:, 0]
        return self.classifier(features)


class RETFound5Class(nn.Module):
    """RETFound with 5-class severity grading head."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.classifier = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Dropout(0.1),
            nn.Linear(1024, 5)
        )

    def forward(self, x):
        features = self.base(x).last_hidden_state[:, 0]
        return self.classifier(features)

    def predict_binary(self, x):
        """Derive binary referral probability from 5-class output."""
        probs = torch.softmax(self.forward(x), dim=-1)
        return probs[:, 2] + probs[:, 3] + probs[:, 4]


class DecorrelationLoss(nn.Module):
    """Combined classification + error decorrelation loss.

    Penalizes correlation between the current model's errors and the
    primary model's errors, encouraging complementary failure modes.
    """

    def __init__(self, alpha=0.08):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.alpha = alpha

    def forward(self, logits, labels, primary_probs):
        cls_loss = self.bce(logits, labels).mean()
        probs = torch.sigmoid(logits)
        secondary_error = torch.abs(probs - labels)
        primary_error = torch.abs(primary_probs - labels)
        se_centered = secondary_error - secondary_error.mean()
        pe_centered = primary_error - primary_error.mean()
        corr = (se_centered * pe_centered).mean() / (
            se_centered.std() * pe_centered.std() + 1e-8
        )
        decorr_penalty = torch.clamp(corr, min=0.0)
        total = cls_loss + self.alpha * decorr_penalty
        return total, cls_loss.item(), decorr_penalty.item()


def build_timm_model(architecture, num_classes, pretrained=True, device="cuda"):
    """Build a standard timm model."""
    model = timm.create_model(architecture, pretrained=pretrained, num_classes=num_classes)
    return model.to(device)


def build_retfound_model(task, device="cuda"):
    """Build a RETFound model with appropriate classification head."""
    from transformers import AutoModel
    base = AutoModel.from_pretrained("iszt/RETFound_mae_meh")
    if task == "5class":
        model = RETFound5Class(base)
    else:
        model = RETFoundBinary(base)
    return model.to(device)


def get_retfound_param_groups(model, base_lr=5e-5, decay_factor=0.65):
    """Create parameter groups with layer-wise learning rate decay for RETFound."""
    param_groups = []
    num_layers = len(model.base.encoder.layer)
    for i, layer in enumerate(model.base.encoder.layer):
        decay = decay_factor ** (num_layers - i)
        param_groups.append({
            "params": layer.parameters(),
            "lr": base_lr * decay
        })
    param_groups.append({
        "params": model.base.embeddings.parameters(),
        "lr": base_lr * decay_factor ** num_layers
    })
    param_groups.append({
        "params": model.classifier.parameters(),
        "lr": base_lr * 10
    })
    return param_groups
