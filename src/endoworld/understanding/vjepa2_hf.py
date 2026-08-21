"""Wrapper around the official V-JEPA 2 weights (HuggingFace transformers).

Provides encode / encode_dense / encode_temporal. Frozen by default; last encoder
blocks can be unfrozen for endoscopic domain adaptation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_MODEL = "facebook/vjepa2-vitl-fpc64-256"

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def _encoder_blocks(model: nn.Module) -> list[nn.Module]:
    for attr in ("encoder", "vision_model", "vit"):
        enc = getattr(model, attr, None)
        if enc is None:
            continue
        for layer_name in ("layer", "layers", "blocks"):
            layers = getattr(enc, layer_name, None)
            if layers is not None and len(list(layers)) > 0:
                return list(layers)
    # fallback: collect modules named *block* / *layer*
    found = [m for n, m in model.named_modules()
             if n.endswith("layer") or "encoder.layer" in n]
    return found


class VJEPA2Encoder(nn.Module):
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None,
                 image_size: int = 256, freeze: bool = True, unfreeze_last: int = 0):
        super().__init__()
        from transformers import AutoModel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.image_size = image_size
        self.embed_dim = getattr(self.model.config, "hidden_size", None)
        self.tubelet = getattr(self.model.config, "tubelet_size", 2)
        self.patch = getattr(self.model.config, "patch_size", 16)
        for p in self.model.parameters():
            p.requires_grad_(False)
        if unfreeze_last > 0:
            blocks = _encoder_blocks(self.model)
            for blk in blocks[-unfreeze_last:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            freeze = False
        if freeze:
            self.model.eval()
        else:
            self.model.train()

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _prep(self, clip: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = clip.shape
        if (h, w) != (self.image_size, self.image_size):
            clip = torch.nn.functional.interpolate(
                clip.reshape(b * t, c, h, w), size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False).reshape(
                    b, t, c, self.image_size, self.image_size)
        clip = (clip - _MEAN.to(clip)) / _STD.to(clip)
        return clip

    def _tokens(self, clip: torch.Tensor) -> torch.Tensor:
        x = self._prep(clip.to(self.device).float())
        trainable = any(p.requires_grad for p in self.model.parameters())
        if trainable and self.training:
            return self.model(pixel_values_videos=x).last_hidden_state
        with torch.no_grad():
            return self.model(pixel_values_videos=x).last_hidden_state

    def encode(self, clip: torch.Tensor) -> torch.Tensor:
        """Pooled clip embedding (B, D)."""
        return self._tokens(clip).mean(dim=1)

    def encode_dense(self, clip: torch.Tensor) -> torch.Tensor:
        """Spatio-temporal tokens (B, t, hw, D) — no spatial pooling."""
        tokens = self._tokens(clip)
        b, n, d = tokens.shape
        t = clip.shape[1] // self.tubelet
        if t <= 0 or n % t != 0:
            return tokens.unsqueeze(1)
        hw = n // t
        return tokens.view(b, t, hw, d)

    def encode_temporal(self, clip: torch.Tensor) -> torch.Tensor:
        """Per-timestep latent sequence (B, t, D), pooled over spatial tokens."""
        return self.encode_dense(clip).mean(dim=2)
