"""CholecT50 supervised fine-tuning with dense MIL heads, official 5-fold CV.

Closes the probe-vs-SOTA supervision gap honestly: same frozen V-JEPA 2
backbone, last K blocks unfrozen, instrument presence via per-token MIL
max-pooling and phase via attention pooling, trained end-to-end on the
official five-fold video split. This is the same-supervision comparison to
task-specific models such as RDV.

    python -m endoworld.eval.cholect50_finetune_mil --unfreeze-last 2 --epochs 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from endoworld.data.cholect50 import N_INSTR, N_PHASE, CHOLECT50_CV_FOLDS
from endoworld.eval.cholect50_probe import build_clips
from endoworld.eval.linear_probe import average_precision


class MILHead(torch.nn.Module):
    """Per-token instrument logits (max-pooled) + attention-pooled phase."""

    def __init__(self, dim: int):
        super().__init__()
        self.token_head = torch.nn.Linear(dim, N_INSTR)
        self.phase_query = torch.nn.Parameter(torch.randn(dim) * 0.02)
        self.phase_head = torch.nn.Linear(dim, N_PHASE)

    def forward(self, tokens):  # (B, N, D)
        instrument = self.token_head(tokens).amax(dim=1)
        weights = torch.softmax(tokens @ self.phase_query / tokens.size(-1) ** 0.5, dim=1)
        phase = self.phase_head((weights.unsqueeze(-1) * tokens).sum(dim=1))
        return phase, instrument


@torch.no_grad()
def evaluate(enc, head, X, yphase, yinstr, idx, device, batch):
    enc.eval()
    head.eval()
    phases, instruments = [], []
    for i in range(0, len(idx), batch):
        sl = idx[i:i + batch]
        tokens = enc.encode_dense(X[sl].to(device).float()).mean(dim=1)
        ph, inst = head(tokens)
        phases.append(ph.argmax(1).cpu())
        instruments.append(torch.sigmoid(inst).cpu())
    pred_phase = torch.cat(phases)
    scores = torch.cat(instruments).numpy()
    yte = yinstr[idx].numpy()
    acc = float((pred_phase == yphase[idx]).float().mean())
    mAP = float(np.nanmean([
        average_precision(scores[:, c], yte[:, c]) for c in range(N_INSTR)]))
    return acc, mAP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/CholecT50/CholecT50")
    parser.add_argument("--vjepa2-id", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-per-video", type=int, default=24)
    parser.add_argument("--unfreeze-last", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/vjepa2_adapted/cholect50_finetune_mil_crossval.json")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from endoworld.understanding.vjepa2_hf import VJEPA2Encoder

    X, Yphase, Yinstr, vids = build_clips(
        args.root, args.clip_len, args.stride, 256, args.max_per_video)
    if X is None:
        raise RuntimeError("no CholecT50 clips built")
    print(f"[ft-mil] {len(X)} clips / {len(set(vids))} videos", flush=True)

    folds = []
    for fold, test_videos in CHOLECT50_CV_FOLDS.items():
        torch.manual_seed(args.seed)
        enc = VJEPA2Encoder(args.vjepa2_id, device=device,
                            unfreeze_last=args.unfreeze_last)
        head = MILHead(enc.embed_dim).to(device)
        params = list(head.parameters()) + enc.trainable_parameters()
        optimizer = torch.optim.AdamW([
            {"params": head.parameters(), "lr": args.head_lr},
            {"params": enc.trainable_parameters(), "lr": args.lr},
        ], weight_decay=0.01)
        test_set = set(test_videos)
        tr = [i for i, v in enumerate(vids) if v not in test_set]
        te = [i for i, v in enumerate(vids) if v in test_set]
        for epoch in range(args.epochs):
            enc.train()
            head.train()
            order = torch.randperm(len(tr))
            running, nb = 0.0, 0
            for i in range(0, len(tr), args.batch):
                sl = [tr[j] for j in order[i:i + args.batch]]
                xb = X[sl].to(device).float()
                tokens = enc.encode_dense(xb).mean(dim=1)
                ph, inst = head(tokens)
                loss = (
                    F.cross_entropy(ph, Yphase[sl].to(device))
                    + F.binary_cross_entropy_with_logits(inst, Yinstr[sl].to(device))
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                running += loss.item()
                nb += 1
            acc, mAP = evaluate(enc, head, X, Yphase, Yinstr,
                                torch.tensor(te), device, args.batch)
            print(f"[{fold} epoch {epoch + 1}] loss={running/max(nb,1):.4f} "
                  f"test acc={acc:.3f} mAP={mAP:.3f}", flush=True)
        folds.append({"fold": fold, "phase_acc": acc, "instrument_mAP": mAP})
        del enc
        torch.cuda.empty_cache()
    report = {
        "protocol": "CholecT50 official five-fold CV; V-JEPA2 last-2-block "
                    "fine-tune + dense MIL/attention heads; same supervision as RDV",
        "unfreeze_last": args.unfreeze_last,
        "epochs_per_fold": args.epochs,
        "phase_acc": float(np.mean([f["phase_acc"] for f in folds])),
        "phase_std_across_folds": float(np.std([f["phase_acc"] for f in folds])),
        "instrument_mAP": float(np.mean([f["instrument_mAP"] for f in folds])),
        "instrument_mAP_std_across_folds": float(np.std([f["instrument_mAP"] for f in folds])),
        "folds": folds,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "folds"}, indent=2))


if __name__ == "__main__":
    main()
