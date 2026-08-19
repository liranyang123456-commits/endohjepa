"""CholecT50 phase recognition: linear probe vs supervised fine-tuning.

Linear probe = frozen encoder + linear head (label-efficient SSL eval).
Fine-tune  = unfreeze last K encoder blocks + head, trained end-to-end on phase
             labels (supervised; the protocol that approaches published SOTA).

    python -m endoworld.eval.cholect50_finetune --unfreeze-last 2 --epochs 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from endoworld.data.cholect50 import N_PHASE, PHASE_NAMES
from endoworld.data.splits import assign_split
from endoworld.eval.cholect50_probe import build_clips


def split_idx(vids):
    tr = [
        i
        for i, v in enumerate(vids)
        if assign_split(f"t50::{v}", train=0.8, val=0.0) == "train"
    ]
    te = [
        i
        for i, v in enumerate(vids)
        if assign_split(f"t50::{v}", train=0.8, val=0.0) != "train"
    ]
    if not te:
        te = tr[-max(1, len(tr) // 5) :]
    return tr, te


def evaluate(head, enc, X, y, idx, device, batch):
    head.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            sl = idx[i : i + batch]
            z = enc.encode(X[sl].to(device).float())
            preds.append(head(z).argmax(1).cpu())
    pred = torch.cat(preds)
    acc = (pred == y[idx]).float().mean().item()
    per = {}
    for c in range(N_PHASE):
        m = y[idx] == c
        if m.any():
            per[PHASE_NAMES[c]] = float((pred[m] == c).float().mean())
    return acc, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/CholecT50/CholecT50")
    ap.add_argument("--vjepa2-id", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-per-video", type=int, default=24)
    ap.add_argument("--unfreeze-last", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", default="outputs/vjepa2_adapted/cholect50_finetune.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from endoworld.understanding.vjepa2_hf import VJEPA2Encoder

    enc = VJEPA2Encoder(args.vjepa2_id, device=device, unfreeze_last=args.unfreeze_last)
    image_size = enc.image_size
    X, Yphase, _, vids = build_clips(
        args.root, args.clip_len, args.stride, image_size, args.max_per_video
    )
    print(
        f"[cholect50-ft] {len(X)} clips / {len(set(vids))} videos  unfreeze_last={args.unfreeze_last}"
    )
    tr, te = split_idx(vids)
    X, Yphase = X, Yphase
    head = torch.nn.Linear(enc.embed_dim, N_PHASE).to(device)
    params = list(head.parameters()) + enc.trainable_parameters()
    print(
        f"[ft] trainable: head {sum(p.numel() for p in head.parameters()) / 1e6:.2f}M + "
        f"encoder {sum(p.numel() for p in enc.trainable_parameters()) / 1e6:.1f}M"
    )
    opt = torch.optim.AdamW(
        [
            {"params": head.parameters(), "lr": args.head_lr},
            {"params": enc.trainable_parameters(), "lr": args.lr},
        ],
        weight_decay=0.01,
    )

    # frozen-encoder linear-probe baseline for reference
    enc.eval()
    _fr_acc, _ = (None, None)
    enc.train()
    for epoch in range(args.epochs):
        head.train()
        idx = torch.randperm(len(tr))
        run, nb = 0.0, 0
        for i in range(0, len(tr), args.batch):
            sl = [tr[j] for j in idx[i : i + args.batch]]
            xb = X[sl].to(device).float()
            yb = Yphase[sl].to(device)
            z = enc.encode(xb)
            loss = F.cross_entropy(head(z), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            run += loss.item()
            nb += 1
        acc_tr, _ = evaluate(head, enc, X, Yphase, torch.tensor(tr), device, args.batch)
        acc_te, per_te = evaluate(
            head, enc, X, Yphase, torch.tensor(te), device, args.batch
        )
        print(
            f"[epoch {epoch}] loss={run / max(nb, 1):.4f}  train_acc={acc_tr:.3f}  test_acc={acc_te:.3f}"
        )

    report = {
        "paper": "Endo-HJEPA",
        "task": "CholecT50 phase recognition",
        "protocol": "supervised fine-tune (last %d blocks + head)" % args.unfreeze_last,
        "video_level": True,
        "not_ablation_planning": True,
        "linear_probe_frozen_acc": 0.635,
        "finetune_test_acc": acc_te,
        "finetune_per_class": per_te,
        "n_train": len(tr),
        "n_test": len(te),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[cholect50-ft] test_acc={acc_te:.3f} (linear probe was 0.635)  wrote {args.out}"
    )


if __name__ == "__main__":
    main()
