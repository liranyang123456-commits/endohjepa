"""STIR start/end point-set consistency on L1 tokens (no pixel regression).

    python -m endoworld.eval.stir_experiment --encoder vjepa2
Optional short unfreeze: --unfreeze-last 1 --epochs 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _pad_points(pts: torch.Tensor, n: int) -> torch.Tensor:
    if pts.numel() == 0:
        return torch.zeros(n, 2)
    if pts.size(0) >= n:
        return pts[:n]
    pad = pts[-1:].expand(n - pts.size(0), 2)
    return torch.cat([pts, pad], 0)


@torch.no_grad()
def evaluate(enc, seqs, image_size: int, device: str, limit: int) -> dict:
    from endoworld.data.stir_tracks import load_stir_clip, stir_clip_tensors
    from endoworld.understanding.l1_regularizers import stir_endpoint_consistency
    rows, vals = [], []
    for seq in seqs[:limit]:
        clip = load_stir_clip(seq)
        if clip is None:
            continue
        packed = stir_clip_tensors(clip, image_size, n_frames=8)
        if packed is None:
            continue
        frames, p0, p1 = packed
        z = enc.encode_dense(frames.unsqueeze(0).to(device).float())
        n = max(len(p0), len(p1), 1)
        p0b = _pad_points(p0, n).unsqueeze(0).to(device)
        p1b = _pad_points(p1, n).unsqueeze(0).to(device)
        loss = stir_endpoint_consistency(z, p0b, p1b, image_size)
        val = float(loss.item())
        vals.append(val)
        rows.append({
            "seq": str(clip.seq_dir),
            "n_start": int(len(clip.points_start)),
            "n_end": int(len(clip.points_end)),
            "n_frames": len(clip.frames),
            "chamfer": val,
        })
    return {
        "n_eval": len(vals),
        "mean_chamfer": float(sum(vals) / len(vals)) if vals else None,
        "rows": rows,
    }


def finetune(enc, seqs, image_size: int, device: str, epochs: int, limit: int, lr: float) -> list[float]:
    from endoworld.data.stir_tracks import load_stir_clip, stir_clip_tensors
    from endoworld.understanding.l1_regularizers import stir_endpoint_consistency
    params = enc.trainable_parameters() if hasattr(enc, "trainable_parameters") else []
    if not params:
        return []
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    hist = []
    batches = []
    for seq in seqs[:limit]:
        clip = load_stir_clip(seq)
        if clip is None:
            continue
        packed = stir_clip_tensors(clip, image_size, n_frames=8)
        if packed is not None:
            batches.append(packed)
    enc.train()
    for epoch in range(epochs):
        run = 0.0
        n = 0
        for frames, p0, p1 in batches:
            z = enc.encode_dense(frames.unsqueeze(0).to(device).float())
            k = max(len(p0), len(p1), 1)
            loss = stir_endpoint_consistency(
                z, _pad_points(p0, k).unsqueeze(0).to(device),
                _pad_points(p1, k).unsqueeze(0).to(device), image_size)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += float(loss.item())
            n += 1
        hist.append(run / max(n, 1))
        print(f"[stir-ft {epoch}] chamfer={hist[-1]:.4f}")
    enc.eval()
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["vjepa2", "scratch"], default="vjepa2")
    ap.add_argument("--vjepa2-id", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--scratch-ckpt", default="outputs/vjepa_l1/vjepa_l1_adapt.pt")
    ap.add_argument("--stir", default="datasets/STIR")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--unfreeze-last", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out", default="outputs/endohjepa_vjepa2/stir_experiment.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from endoworld.data.stir_tracks import find_stir_sequences
    from endoworld.understanding.encoders import load_any_encoder
    if args.encoder == "vjepa2" and args.unfreeze_last > 0:
        from endoworld.understanding.vjepa2_hf import VJEPA2Encoder
        enc = VJEPA2Encoder(args.vjepa2_id, device=device, unfreeze_last=args.unfreeze_last)
        image_size = enc.image_size
    else:
        enc, _, image_size, _ = load_any_encoder(
            args.encoder, device, args.vjepa2_id, args.scratch_ckpt)
    seqs = find_stir_sequences(args.stir)
    before = evaluate(enc, seqs, image_size, device, args.limit)
    hist = []
    if args.epochs > 0 and args.unfreeze_last > 0:
        hist = finetune(enc, seqs, image_size, device, args.epochs, args.limit, args.lr)
    after = evaluate(enc, seqs, image_size, device, args.limit) if hist else before
    report = {
        "paper": "Endo-HJEPA",
        "not_ct_ablation_planning": True,
        "encoder": args.encoder,
        "main_table_ok": args.encoder == "vjepa2",
        "n_sequences": len(seqs),
        "before": {"n_eval": before["n_eval"], "mean_chamfer": before["mean_chamfer"]},
        "after": {"n_eval": after["n_eval"], "mean_chamfer": after["mean_chamfer"]},
        "finetune_losses": hist,
        "rows": after["rows"],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[stir] mean chamfer={after['mean_chamfer']}  wrote {args.out}")


if __name__ == "__main__":
    main()
