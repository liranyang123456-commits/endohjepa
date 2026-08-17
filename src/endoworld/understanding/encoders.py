"""Load scratch V-JEPA or official V-JEPA2 for eval / adapt."""
from __future__ import annotations

import os


def load_any_encoder(encoder: str, device: str, vjepa2_id: str = "facebook/vjepa2-vitl-fpc64-256",
                     scratch_ckpt: str = "outputs/vjepa_l1/vjepa_l1_adapt.pt"):
    if encoder == "vjepa2":
        from endoworld.understanding.vjepa2_hf import VJEPA2Encoder
        enc = VJEPA2Encoder(vjepa2_id, device=device)
        return enc, 16, enc.image_size, enc.embed_dim
    from endoworld.understanding.vjepa import VJEPA, VJEPAConfig
    if scratch_ckpt and os.path.isfile(scratch_ckpt):
        import torch
        ck = torch.load(scratch_ckpt, map_location=device, weights_only=False)
        cfg = VJEPAConfig(**ck["cfg"])
        model = VJEPA(cfg).to(device).eval()
        model.load_state_dict(ck["model"])
        for p in model.parameters():
            p.requires_grad_(False)
        return model, cfg.clip_len, cfg.image_size, cfg.embed_dim
    raise FileNotFoundError(f"scratch encoder missing: {scratch_ckpt}")


def load_adapted_vjepa2(ckpt_path: str, device: str):
    """Load a V-JEPA2 encoder fine-tuned by end-to-end domain adaptation.

    The e2e run saves the underlying HF model state_dict under "encoder".
    """
    import torch
    from endoworld.understanding.vjepa2_hf import VJEPA2Encoder, DEFAULT_MODEL
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc = VJEPA2Encoder(blob.get("vjepa2_id", DEFAULT_MODEL), device=device)
    enc.model.load_state_dict(blob["encoder"])
    for p in enc.model.parameters():
        p.requires_grad_(False)
    enc.model.eval()
    return enc, 16, enc.image_size, enc.embed_dim
