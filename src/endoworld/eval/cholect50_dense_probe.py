"""Dense-token MIL probe for CholecT50 instrument presence and phase.

The pooled linear probe averages 256 spatial tokens, washing out small
instruments. This probe keeps the spatial axis: instrument presence uses
multiple-instance learning (per-token logits, max-pooled over tokens), and
phase uses attention pooling over tokens followed by a linear head. Features
are frozen V-JEPA 2 dense tokens averaged over tubelets; the official
five-fold video-level CV and 3 probe seeds match cholect50_probe.py.

    python -m endoworld.eval.cholect50_dense_probe --crossval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from endoworld.data.cholect50 import N_INSTR, N_PHASE, CHOLECT50_CV_FOLDS
from endoworld.eval.cholect50_probe import build_clips, split_indices
from endoworld.eval.linear_probe import average_precision


@torch.no_grad()
def extract_dense(model, X, device, batch=2):
    """(B, T, C, H, W) -> (B, N, D) spatial tokens averaged over tubelets."""
    feats = []
    for i in range(0, len(X), batch):
        dense = model.encode_dense(X[i : i + batch].to(device).float())
        feats.append(dense.mean(dim=1).cpu())
    return torch.cat(feats)


class MILInstrumentProbe(torch.nn.Module):
    """Per-token instrument logits, max-pooled over space (weak localisation)."""

    def __init__(self, dim: int):
        super().__init__()
        self.token_head = torch.nn.Linear(dim, N_INSTR)

    def forward(self, tokens):  # (B, N, D) -> (B, N_INSTR)
        return self.token_head(tokens).amax(dim=1)


class AttentionPhaseProbe(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(dim) * 0.02)
        self.head = torch.nn.Linear(dim, N_PHASE)

    def forward(self, tokens):
        weights = torch.softmax(tokens @ self.query / tokens.size(-1) ** 0.5, dim=1)
        return self.head((weights.unsqueeze(-1) * tokens).sum(dim=1))


def _fit_probe(head_factory, feats, target, task, device, seed, steps=400, lr=0.02):
    torch.manual_seed(seed)
    head = head_factory().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        if task == "phase":
            loss = F.cross_entropy(head(feats[0].to(device)), target[0].to(device))
        else:
            loss = F.binary_cross_entropy_with_logits(
                head(feats[0].to(device)), target[0].to(device)
            )
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head.eval()


def _run_fold(feat, yphase, yinstr, vids, device, test_videos, seeds):
    tr, te = split_indices(vids, False, test_videos=test_videos)
    tr_t, te_t = torch.tensor(tr), torch.tensor(te)
    mu = feat[tr_t].mean(dim=(0, 1), keepdim=True)
    sd = feat[tr_t].std(dim=(0, 1), keepdim=True) + 1e-6
    fz = (feat - mu) / sd
    phase_accs, instr_maps = [], []
    for seed in seeds:
        ph = _fit_probe(
            lambda: AttentionPhaseProbe(feat.size(-1)),
            (fz[tr_t],),
            (yphase[tr_t],),
            "phase",
            device,
            seed,
        )
        with torch.no_grad():
            pred = ph(fz[te_t].to(device)).argmax(1).cpu()
        phase_accs.append(float((pred == yphase[te_t]).float().mean()))
        mil = _fit_probe(
            lambda: MILInstrumentProbe(feat.size(-1)),
            (fz[tr_t],),
            (yinstr[tr_t],),
            "instr",
            device,
            seed,
        )
        with torch.no_grad():
            scores = torch.sigmoid(mil(fz[te_t].to(device))).cpu().numpy()
        yte = yinstr[te_t].numpy()
        instr_maps.append(
            float(
                np.nanmean(
                    [average_precision(scores[:, c], yte[:, c]) for c in range(N_INSTR)]
                )
            )
        )
    return {
        "phase_acc": float(np.mean(phase_accs)),
        "phase_std": float(np.std(phase_accs)),
        "instrument_mAP": float(np.mean(instr_maps)),
        "instrument_mAP_std": float(np.std(instr_maps)),
        "n_train": len(tr),
        "n_test": len(te),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/CholecT50/CholecT50")
    parser.add_argument("--vjepa2-id", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-per-video", type=int, default=24)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--out", default="outputs/vjepa2_adapted/cholect50_dense_probe_crossval.json"
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from endoworld.understanding.encoders import load_any_encoder

    enc, _, _, _ = load_any_encoder("vjepa2", device, args.vjepa2_id, "")
    X, Yphase, Yinstr, vids = build_clips(
        args.root, args.clip_len, args.stride, 256, args.max_per_video
    )
    if X is None:
        raise RuntimeError("no CholecT50 clips built")
    print(f"[dense-probe] {len(X)} clips / {len(set(vids))} videos", flush=True)
    feat = extract_dense(enc, X, device)
    print(f"[dense-probe] features {tuple(feat.shape)}", flush=True)
    folds = []
    for fold, test_videos in CHOLECT50_CV_FOLDS.items():
        row = _run_fold(
            feat,
            Yphase,
            Yinstr,
            vids,
            device,
            test_videos,
            seeds=tuple(range(args.seeds)),
        )
        row["fold"] = fold
        folds.append(row)
        print(
            f"[dense-probe] {fold}: phase={row['phase_acc']:.3f} "
            f"mAP={row['instrument_mAP']:.3f}",
            flush=True,
        )
    report = {
        "protocol": "CholecT50 official five-fold CV; frozen V-JEPA2 dense tokens; "
        "MIL max-pool instrument probe + attention-pool phase probe",
        "n_clips": len(X),
        "seeds": args.seeds,
        "phase_acc": float(np.mean([f["phase_acc"] for f in folds])),
        "phase_std_across_folds": float(np.std([f["phase_acc"] for f in folds])),
        "instrument_mAP": float(np.mean([f["instrument_mAP"] for f in folds])),
        "instrument_mAP_std_across_folds": float(
            np.std([f["instrument_mAP"] for f in folds])
        ),
        "folds": folds,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "folds"}, indent=2))


if __name__ == "__main__":
    main()
