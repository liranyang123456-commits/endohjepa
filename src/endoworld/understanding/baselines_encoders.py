"""External baseline encoders for representation comparison (the SOTA table).

- VideoMAE (self-supervised video, Kinetics): strong video SSL baseline.
- ImageNet-supervised ViT: per-frame supervised features pooled over time.

Both expose encode(clip (B,T,C,H,W) in [0,1]) -> (B, D) pooled, and .image_size.

    python -m endoworld.understanding.baselines_encoders --smoke
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def _norm(clip, mean, std):
    return (clip - mean.to(clip) / 1.0) / std.to(clip)


class VideoMAEEncoder(nn.Module):
    """MCG-NJU/videomae-base: self-supervised video model (Kinetics-400)."""

    def __init__(
        self, model_id: str = "MCG-NJU/videomae-base", device: str | None = None
    ):
        super().__init__()
        from transformers import VideoMAEModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VideoMAEModel.from_pretrained(model_id).to(self.device).eval()
        cfg = self.model.config
        self.image_size = cfg.image_size  # 224
        self.num_frames = cfg.num_frames  # 16
        self.embed_dim = cfg.hidden_size  # 768
        self.tubelet = cfg.tubelet_size  # 2
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _prep(self, clip):
        b, t, c, h, w = clip.shape
        if (h, w) != (self.image_size, self.image_size):
            clip = torch.nn.functional.interpolate(
                clip.reshape(b * t, c, h, w),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(b, t, c, self.image_size, self.image_size)
        return (clip - _MEAN.to(clip)) / _STD.to(clip)

    @torch.no_grad()
    def encode(self, clip):
        x = self._prep(clip.to(self.device).float())
        # VideoMAE wants (batch, num_frames, channels, H, W) == our (B,T,C,H,W)
        out = self.model(pixel_values=x).last_hidden_state
        return out.mean(dim=1)


class ImageNetViTEncoder(nn.Module):
    """google/vit-base-patch16-224: supervised ImageNet features, pooled over time."""

    def __init__(
        self, model_id: str = "google/vit-base-patch16-224", device: str | None = None
    ):
        super().__init__()
        from transformers import ViTModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViTModel.from_pretrained(model_id).to(self.device).eval()
        self.image_size = 224
        self.embed_dim = self.model.config.hidden_size  # 768
        self.num_frames = 1
        self.tubelet = 1
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, clip):
        b, t, c, h, w = clip.shape
        x = clip.to(self.device).float().reshape(b * t, c, h, w)  # always 4D per-frame
        if (h, w) != (self.image_size, self.image_size):
            x = torch.nn.functional.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        x = (x - _MEAN.to(x)[:, 0]) / _STD.to(x)[:, 0]
        out = self.model(pixel_values=x).last_hidden_state  # (B*T, 1+P, D)
        feat = out[:, 0]  # CLS token
        return feat.reshape(b, t, -1).mean(dim=1)  # pool over time


_HF_CACHE = Path.home() / ".cache" / "endoworld_hf"


def _download_via_get(model_id: str, files: list[str]) -> str:
    """Download model files via direct GET (hf-mirror), bypassing the flaky HEAD check.

    The huggingface_hub HEAD pre-flight fails on this network even though direct GET
    works, so we fetch the resolve URLs straight into a local dir.
    """
    import urllib.request

    out = _HF_CACHE / model_id.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)
    base = "https://hf-mirror.com"
    for f in files:
        dest = out / f
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        for host in (base, "https://huggingface.co"):
            url = f"{host}/{model_id}/resolve/main/{f}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with (
                    urllib.request.urlopen(req, timeout=120) as r,
                    open(dest, "wb") as w,
                ):
                    w.write(r.read())
                break
            except Exception:
                continue
        if not dest.exists():
            raise OSError(f"could not download {model_id}/{f}")
    return str(out)


def _load_local(model_id: str, files=("config.json", "model.safetensors")):
    """Download via direct GET and load with AutoModel from the local dir.

    Tries model.safetensors then pytorch_model.bin for the weights.
    """
    from transformers import AutoModel

    try:
        local = _download_via_get(model_id, ["config.json", "model.safetensors"])
    except OSError:
        local = _download_via_get(model_id, ["config.json", "pytorch_model.bin"])
    return AutoModel.from_pretrained(local)


class DINOv2Encoder(nn.Module):
    """facebook/dinov2-base: strong self-supervised image features, pooled over time."""

    def __init__(
        self, model_id: str = "facebook/dinov2-base", device: str | None = None
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _load_local(model_id).to(self.device).eval()
        self.image_size = 224
        self.embed_dim = self.model.config.hidden_size  # 768
        self.num_frames = 1
        self.tubelet = 1
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, clip):
        b, t, c, h, w = clip.shape
        x = clip.to(self.device).float().reshape(b * t, c, h, w)
        if (h, w) != (self.image_size, self.image_size):
            x = torch.nn.functional.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        x = (x - _MEAN.to(x)[:, 0]) / _STD.to(x)[:, 0]
        out = self.model(pixel_values=x).last_hidden_state
        feat = out[:, 0]  # CLS
        return feat.reshape(b, t, -1).mean(dim=1)


class TimeSformerEncoder(nn.Module):
    """facebook/timesformer-base-finetuned-k400: divided space-time video Transformer."""

    def __init__(
        self, model_id: str = "facebook/timesformer-base-finetuned-k400", device=None
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _load_local(model_id).to(self.device).eval()
        cfg = self.model.config
        self.image_size = getattr(cfg, "image_size", 224)
        self.num_frames = getattr(cfg, "num_frames", 8)
        self.embed_dim = cfg.hidden_size
        self.tubelet = 1
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, clip):
        b, t, c, h, w = clip.shape
        x = clip.to(self.device).float()
        if (h, w) != (self.image_size, self.image_size):
            x = torch.nn.functional.interpolate(
                x.reshape(b * t, c, h, w),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(b, t, c, self.image_size, self.image_size)
        x = (x - _MEAN.to(x)) / _STD.to(x)
        # TimeSformer wants (batch, frames, C, H, W) == our (B,T,C,H,W)
        out = self.model(pixel_values=x).last_hidden_state
        return out[:, 0] if out.dim() == 3 else out.mean(dim=1)


class ViViTEncoder(nn.Module):
    """google/vivit-b-16x2-kinetics400: factorised video Transformer."""

    def __init__(self, model_id: str = "google/vivit-b-16x2-kinetics400", device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _load_local(model_id).to(self.device).eval()
        cfg = self.model.config
        self.image_size = getattr(cfg, "image_size", 224)
        self.num_frames = getattr(cfg, "num_frames", 32)
        self.embed_dim = cfg.hidden_size
        self.tubelet = getattr(cfg, "tubelet_size", 2)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, clip):
        b, t, c, h, w = clip.shape
        x = clip.to(self.device).float()
        # ViViT expects a fixed number of frames; resample temporally to match
        if t != self.num_frames:
            idx = torch.linspace(0, t - 1, self.num_frames).round().long().to(x.device)
            x = x[:, idx]
            t = self.num_frames
        if (h, w) != (self.image_size, self.image_size):
            x = torch.nn.functional.interpolate(
                x.reshape(b * t, c, h, w),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(b, t, c, self.image_size, self.image_size)
        x = (x - _MEAN.to(x)) / _STD.to(x)
        out = self.model(pixel_values=x).last_hidden_state
        return out.mean(dim=1)


def load_baseline(name: str, device: str):
    if name == "videomae":
        return VideoMAEEncoder(device=device)
    if name == "imagenet":
        return ImageNetViTEncoder(device=device)
    if name == "dinov2":
        return DINOv2Encoder(device=device)
    if name == "timesformer":
        return TimeSformerEncoder(device=device)
    if name == "vivit":
        return ViViTEncoder(device=device)
    raise ValueError(name)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        for name in ("videomae", "imagenet"):
            enc = load_baseline(name, "cuda" if torch.cuda.is_available() else "cpu")
            x = torch.rand(1, 16, 3, 224, 224)
            f = enc.encode(x)
            print(name, "->", tuple(f.shape), "dim", enc.embed_dim)
