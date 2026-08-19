"""EndoVis instrument-mask experiment: token motion + weighted L1.

    python -m endoworld.eval.instrument_mask_exp --encoder vjepa2
Do not cite CholecSeg8k clip-leaky 0.992 mAP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _pool_mask(inst: torch.Tensor, hw: int) -> torch.Tensor:
    """inst (T,H,W) → (T', hw) matching tubelet-pooled spatial tokens."""
    t, h, w = inst.shape
    side = int(hw**0.5)
    if side * side != hw:
        return inst.mean(dim=0).reshape(-1)[:hw].unsqueeze(0).expand(max(t // 2, 1), -1)
    x = F.interpolate(inst.unsqueeze(1), size=(side, side), mode="area").squeeze(1)
    # match tubelet=2 by averaging pairs of frames
    t2 = (x.size(0) // 2) * 2
    if t2 >= 2:
        x = x[:t2].reshape(t2 // 2, 2, side * side).mean(1)
    else:
        x = x.reshape(x.size(0), -1)
    return x


@torch.no_grad()
def token_motion(
    enc, root: str, split: str, clip_len: int, image_size: int, device: str, limit: int
) -> dict:
    from endoworld.data.endovis_masks import iter_endovis_clips

    inst_m, bg_m, fracs = [], [], []
    n = 0
    for seq, clip, inst in iter_endovis_clips(root, split, clip_len, image_size, limit):
        z = enc.encode_dense(clip.unsqueeze(0).to(device).float())[0]  # (T,N,D)
        if z.size(0) < 2:
            continue
        dz = (z[1:] - z[:-1]).pow(2).mean(-1)  # (T-1, N)
        w = _pool_mask(inst, z.size(1)).to(device)
        w = w[: dz.size(0)]
        inst_w = w.clamp(0, 1)
        bg_w = 1.0 - inst_w
        if float(inst_w.sum()) > 0:
            inst_m.append(float((dz * inst_w).sum() / inst_w.sum()))
        if float(bg_w.sum()) > 0:
            bg_m.append(float((dz * bg_w).sum() / bg_w.sum()))
        fracs.append(float(inst.mean()))
        n += 1
        print(f"[mask] {seq} inst_frac={fracs[-1]:.3f}")
    return {
        "n_clips": n,
        "mean_instrument_fraction": float(np.mean(fracs)) if fracs else None,
        "token_mse_instrument": float(np.mean(inst_m)) if inst_m else None,
        "token_mse_background": float(np.mean(bg_m)) if bg_m else None,
        "ratio_inst_over_bg": (
            float(np.mean(inst_m) / max(np.mean(bg_m), 1e-8))
            if inst_m and bg_m
            else None
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["vjepa2", "scratch"], default="vjepa2")
    ap.add_argument("--vjepa2-id", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--scratch-ckpt", default="outputs/vjepa_l1/vjepa_l1_adapt.pt")
    ap.add_argument("--root", default="datasets/endovis2017_full/endovis2017")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument(
        "--out", default="outputs/endohjepa_vjepa2/instrument_mask_exp.json"
    )
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from endoworld.understanding.encoders import load_any_encoder

    enc, clip_len, image_size, _ = load_any_encoder(
        args.encoder, device, args.vjepa2_id, args.scratch_ckpt
    )
    motion = token_motion(
        enc, args.root, args.split, clip_len, image_size, device, args.limit
    )
    report = {
        "paper": "Endo-HJEPA",
        "not_ct_ablation_planning": True,
        "not_cholecseg8k_0992": True,
        "encoder": args.encoder,
        "main_table_ok": args.encoder == "vjepa2",
        "token_motion": motion,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(motion, indent=2))
    print(f"[instrument-mask] wrote {args.out}")


if __name__ == "__main__":
    main()
