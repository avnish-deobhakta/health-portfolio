"""
Model definitions for health-portfolio ensemble analysis.

Includes standard timm models (binary and 5-class) and RETFound wrappers.
"""

import torch
import torch.nn as nn
import timm


def _infer_hidden_size(base):
    """Infer hidden size from a Hugging Face vision backbone."""
    if hasattr(base, "config") and hasattr(base.config, "hidden_size"):
        return base.config.hidden_size
    # Fallback for unexpected backbones
    sample = getattr(base, "embeddings", None)
    if sample is not None and hasattr(sample, "patch_embeddings"):
        proj = getattr(sample.patch_embeddings, "projection", None)
        if proj is not None and hasattr(proj, "out_channels"):
            return proj.out_channels
    raise ValueError("Could not infer hidden size from RETFound backbone")


class BinaryClassifier(nn.Module):
    """Standard timm model with binary classification head."""

    def __init__(self, model_name="densenet121", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        self.classifier = nn.Linear(num_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(-1)


class FiveClassClassifier(nn.Module):
    """Standard timm model with 5-class severity grading head."""

    def __init__(self, model_name="densenet121", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        self.classifier = nn.Linear(num_features, 5)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def predict_proba(self, x):
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def predict_binary(self, x):
        """Collapse 5-class probabilities to referable probability.

        Referable = grades 2, 3, 4 (moderate NPDR, severe NPDR, PDR).
        """
        probs = self.predict_proba(x)
        referable_prob = probs[:, 2:].sum(dim=-1)
        return referable_prob


class RETFoundBinary(nn.Module):
    """RETFound with binary classification head."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        hidden_size = _infer_hidden_size(base)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        outputs = self.base(x)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_token)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(-1)


class RETFound5Class(nn.Module):
    """RETFound with 5-class severity grading head."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        hidden_size = _infer_hidden_size(base)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 5)
        )

    def forward(self, x):
        outputs = self.base(x)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_token)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def predict_binary(self, x):
        """Collapse 5-class probabilities to referable probability."""
        probs = self.predict_proba(x)
        referable_prob = probs[:, 2:].sum(dim=-1)
        return referable_prob


class RETFoundAdversarial(nn.Module):
    """RETFound with adversarial decorrelation training support.

    During training, adds a correlation penalty between this model's
    errors and a primary model's errors:
        L = L_task + alpha * corr(errors_primary, errors_adversarial)
    """

    def __init__(self, base, alpha=0.08):
        super().__init__()
        self.base = base
        self.alpha = alpha
        hidden_size = _infer_hidden_size(base)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        outputs = self.base(x)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_token)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(-1)
