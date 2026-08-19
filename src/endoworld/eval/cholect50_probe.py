"""CholecT50 downstream probes: phase recognition + instrument presence.

Video-level split (by VID), frozen vs domain-adapted V-JEPA 2. Supports the
five-video challenge split and the recommended official five-fold CV. These
linear probes are representation diagnostics, not task-specific SOTA models.

    python -m endoworld.eval.cholect50_probe --compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from endoworld.data.cholect50 import (
    N_INSTR,
    N_PHASE,
    PHASE_NAMES,
    CHALLENGE_TEST_VIDS,
    CHOLECT50_CV_FOLDS,
    list_videos,
    load_video_labels,
    video_frames_dir,
)
from endoworld.data.splits import assign_split


def _clip_starts(
    n_frames: int,
    clip_len: int,
    stride: int,
    max_per_video: int,
) -> list[int]:
    """Return evenly spaced starts whose complete strided clip is in bounds."""
    span = (clip_len - 1) * stride + 1
    if n_frames < span or max_per_video < 1:
        return []
    n_valid = n_frames - span + 1
    n_samples = min(max_per_video, max(1, n_valid))
    return np.linspace(0, n_valid - 1, n_samples).astype(int).tolist()


def split_indices(vids, official: bool, test_videos=None):
    if test_videos is not None:
        test_set = set(test_videos)
        tr = [i for i, v in enumerate(vids) if v not in test_set]
        te = [i for i, v in enumerate(vids) if v in test_set]
        return tr, te
    if official:
        tr = [i for i, v in enumerate(vids) if v not in CHALLENGE_TEST_VIDS]
        te = [i for i, v in enumerate(vids) if v in CHALLENGE_TEST_VIDS]
        return tr, te
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
    return tr, te


def build_clips(root, clip_len, stride, image_size, max_per_video):
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    videos_dir = Path(root) / "videos"
    X, Yphase, Yinstr, vids = [], [], [], []
    for vid in list_videos():
        vdir = video_frames_dir(vid, videos_dir)
        if not vdir.is_dir():
            continue
        labels = load_video_labels(vid)
        frames = sorted(vdir.glob("*.png"))
        n = len(frames)
        # Subsample complete strided clips; never clamp or repeat the final frame.
        starts = _clip_starts(n, clip_len, stride, max_per_video)
        for s in starts:
            idxs = [s + i * stride for i in range(clip_len)]
            imgs = []
            ok = True
            for i in idxs:
                try:
                    im = (
                        Image.open(frames[i])
                        .convert("RGB")
                        .resize((image_size, image_size))
                    )
                    imgs.append(np.asarray(im, np.float32) / 255.0)
                except Exception:
                    ok = False
                    break
            if not ok:
                continue
            # label from center frame
            center_idx = idxs[len(idxs) // 2]
            lab = labels.get(center_idx)
            if lab is None or lab["phase"] < 0:
                continue
            X.append(np.stack(imgs).transpose(0, 3, 1, 2))
            Yphase.append(lab["phase"])
            Yinstr.append(lab["instr"])
            vids.append(vid)
    if not X:
        return None, None, None, None
    return (
        torch.from_numpy(np.stack(X)),
        torch.tensor(Yphase, dtype=torch.long),
        torch.from_numpy(np.stack(Yinstr)),
        vids,
    )


@torch.no_grad()
def extract(model, X, device, batch=4):
    feats = []
    for i in range(0, len(X), batch):
        feats.append(model.encode(X[i : i + batch].to(device).float()).cpu())
    return torch.cat(feats)


def probe_phase(
    feat, yphase, vids, device, official=False, seeds=(0, 1, 2), test_videos=None
):
    tr, te = split_indices(vids, official, test_videos)
    if not te:
        te = tr[-max(1, len(tr) // 5) :]
    tr_t, te_t = torch.tensor(tr), torch.tensor(te)
    # Fit normalization on training clips only; using all clips is transductive leakage.
    mu = feat[tr_t].mean(0, keepdim=True)
    sd = feat[tr_t].std(0, keepdim=True) + 1e-6
    fz = (feat - mu) / sd
    accs = []
    per_accum = {}
    for seed in seeds:
        torch.manual_seed(seed)  # reproducible per-seed probe head init
        head = torch.nn.Linear(feat.size(1), N_PHASE).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=0.05, weight_decay=1e-4)
        ytr = yphase[tr_t].to(device)
        for _ in range(300):
            loss = F.cross_entropy(head(fz[tr_t].to(device)), ytr)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = head(fz[te_t].to(device)).argmax(1).cpu()
        accs.append((pred == yphase[te_t]).float().mean().item())
        for c in range(N_PHASE):
            m = yphase[te_t] == c
            if m.any():
                per_accum.setdefault(PHASE_NAMES[c], []).append(
                    float((pred[m] == c).float().mean())
                )
    accs = np.array(accs)
    per = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in per_accum.items()
    }
    return {
        "acc": float(accs.mean()),
        "acc_std": float(accs.std()),
        "accs": accs.tolist(),
        "per_class_acc": per,
        "n_train": len(tr),
        "n_test": len(te),
        "n_seeds": len(seeds),
    }


def probe_instrument(
    feat, yinstr, vids, device, official=False, seeds=(0, 1, 2), test_videos=None
):
    from endoworld.eval.linear_probe import average_precision

    tr, te = split_indices(vids, official, test_videos)
    if not te:
        te = tr[-max(1, len(tr) // 5) :]
    tr_t, te_t = torch.tensor(tr), torch.tensor(te)
    mu = feat[tr_t].mean(0, keepdim=True)
    sd = feat[tr_t].std(0, keepdim=True) + 1e-6
    fz = (feat - mu) / sd
    yte = yinstr[te_t].numpy()
    maps = []
    for seed in seeds:
        torch.manual_seed(seed)
        head = torch.nn.Linear(feat.size(1), N_INSTR).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=0.05, weight_decay=1e-4)
        for _ in range(300):
            loss = F.binary_cross_entropy_with_logits(
                head(fz[tr_t].to(device)), yinstr[tr_t].to(device)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            scores = torch.sigmoid(head(fz[te_t].to(device))).cpu().numpy()
        aps = [average_precision(scores[:, c], yte[:, c]) for c in range(N_INSTR)]
        maps.append(float(np.nanmean(aps)))
    maps = np.array(maps)
    return {
        "mAP": float(maps.mean()),
        "mAP_std": float(maps.std()),
        "mAPs": maps.tolist(),
        "n_seeds": len(seeds),
    }


def probe_crossval(feat, yphase, yinstr, vids, device, seeds=(0, 1, 2)):
    """Official five-fold video CV; features are extracted once, heads per fold."""
    folds = []
    for fold, test_videos in CHOLECT50_CV_FOLDS.items():
        ph = probe_phase(
            feat, yphase, vids, device, seeds=seeds, test_videos=test_videos
        )
        inst = probe_instrument(
            feat, yinstr, vids, device, seeds=seeds, test_videos=test_videos
        )
        folds.append(
            {
                "fold": fold,
                "test_videos": test_videos,
                "phase": ph,
                "instrument": inst,
            }
        )
    phase_vals = np.array([x["phase"]["acc"] for x in folds])
    instr_vals = np.array([x["instrument"]["mAP"] for x in folds])
    return {
        "protocol": "CholecT50 official five-fold cross-validation",
        "n_folds": len(folds),
        "n_probe_seeds_per_fold": len(seeds),
        "phase": {
            "acc": float(phase_vals.mean()),
            "acc_std_across_folds": float(phase_vals.std()),
        },
        "instrument": {
            "mAP": float(instr_vals.mean()),
            "mAP_std_across_folds": float(instr_vals.std()),
        },
        "folds": folds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/CholecT50/CholecT50")
    ap.add_argument("--vjepa2-id", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument(
        "--adapted-ckpt", default="outputs/vjepa2_adapted/vjepa2_adapted.pt"
    )
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-per-video", type=int, default=24)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument(
        "--official-split",
        action="store_true",
        help="test on the official CholecTriplet challenge videos VID68-75",
    )
    ap.add_argument(
        "--crossval",
        action="store_true",
        help="recommended official CholecT50 five-fold CV",
    )
    ap.add_argument(
        "--encoders",
        default="vjepa2-frozen",
        help="comma list: imagenet,videomae,vjepa2-frozen,vjepa2-adapted",
    )
    ap.add_argument("--out", default="outputs/vjepa2_adapted/cholect50_probe.json")
    args = ap.parse_args()
    if args.crossval and args.official_split:
        ap.error("--crossval and --official-split are mutually exclusive")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from endoworld.understanding.encoders import load_any_encoder, load_adapted_vjepa2
    from endoworld.understanding.baselines_encoders import load_baseline

    encoders = {}
    names = [e.strip() for e in args.encoders.split(",") if e.strip()]
    if "vjepa2-frozen" in names or args.compare:
        fr, _, _, _ = load_any_encoder("vjepa2", device, args.vjepa2_id, "")
        encoders["vjepa2-frozen"] = fr
    if "vjepa2-adapted" in names and Path(args.adapted_ckpt).is_file():
        ad, _, _, _ = load_adapted_vjepa2(args.adapted_ckpt, device)
        encoders["vjepa2-adapted"] = ad
    for b in ("imagenet", "videomae", "dinov2", "timesformer", "vivit"):
        if b in names:
            encoders[b] = load_baseline(b, device)
    if (
        args.compare
        and "vjepa2-adapted" not in encoders
        and Path(args.adapted_ckpt).is_file()
    ):
        ad, _, _, _ = load_adapted_vjepa2(args.adapted_ckpt, device)
        encoders["vjepa2-adapted"] = ad

    # build clips once at a shared size; each encoder resizes internally
    X, Yphase, Yinstr, vids = build_clips(
        args.root, args.clip_len, args.stride, 256, args.max_per_video
    )
    if X is None:
        print("[cholect50] no clips built; check frames extracted")
        return
    print(f"[cholect50] {len(X)} clips / {len(set(vids))} videos")

    report = {
        "paper": "Endo-HJEPA",
        "video_level": True,
        "not_ablation_planning": True,
        "official_split": bool(args.official_split),
        "crossval": bool(args.crossval),
        "test_videos": (
            CHOLECT50_CV_FOLDS
            if args.crossval
            else CHALLENGE_TEST_VIDS
            if args.official_split
            else "hash"
        ),
        "n_clips": len(X),
        "n_videos": len(set(vids)),
        "results": {},
    }
    for tag, model in encoders.items():
        feat = extract(model, X, device, batch=4)
        if args.crossval:
            cv = probe_crossval(feat, Yphase, Yinstr, vids, device)
            report["results"][tag] = cv
            print(
                f"[{tag}] CV phase acc={cv['phase']['acc']:.3f}  "
                f"instrument mAP={cv['instrument']['mAP']:.3f}"
            )
        else:
            ph = probe_phase(feat, Yphase, vids, device, args.official_split)
            inst = probe_instrument(feat, Yinstr, vids, device, args.official_split)
            report["results"][tag] = {"phase": ph, "instrument": inst}
            print(
                f"[{tag}] phase acc={ph['acc']:.3f}  instrument mAP={inst['mAP']:.3f}"
            )
    if len(encoders) == 2 and not args.crossval:
        report["adaptation_gain_phase"] = (
            report["results"]["vjepa2-adapted"]["phase"]["acc"]
            - report["results"]["vjepa2-frozen"]["phase"]["acc"]
        )
        report["adaptation_gain_instrument"] = (
            report["results"]["vjepa2-adapted"]["instrument"]["mAP"]
            - report["results"]["vjepa2-frozen"]["instrument"]["mAP"]
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[cholect50] wrote {args.out}")


if __name__ == "__main__":
    main()
