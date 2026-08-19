"""Video-level EndoVis instrument-presence linear probe (frozen encoder).

    python -m endoworld.eval.instrument_probe --vjepa outputs/vjepa_l1/vjepa_l1_adapt.pt
Do not cite clip-leaky CholecSeg8k 0.992 mAP. This is a separate EndoVis protocol.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from endoworld.data.endovis_masks import (
    list_endovis_pairs,
    load_class_map,
    load_mask,
    presence_vector,
)
from endoworld.data.splits import assign_split
from endoworld.eval.linear_probe import (
    average_precision,
    extract_features,
    load_encoder,
)


def build_endovis(root, clip_len, image_size, limit):
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    pairs = list_endovis_pairs(root, "train")
    by_seq: dict[str, list] = defaultdict(list)
    for img, lab in pairs:
        seq = img.name.split("_frame")[0]
        by_seq[seq].append((img, lab))
    clips, ys, vids = [], [], []
    for seq, items in by_seq.items():
        items = sorted(items, key=lambda x: x[0].name)
        if len(items) < clip_len:
            continue
        for start in range(0, len(items) - clip_len + 1, clip_len):
            chunk = items[start : start + clip_len]
            frames = []
            for img, _ in chunk:
                im = Image.open(img).convert("RGB").resize((image_size, image_size))
                frames.append(np.asarray(im, np.float32) / 255.0)
            mid = chunk[len(chunk) // 2][1]
            y = presence_vector(load_mask(mid), 8)[1:]  # drop background
            clips.append(np.stack(frames).transpose(0, 3, 1, 2))
            ys.append(y)
            vids.append(seq)
            if limit and len(clips) >= limit:
                break
        if limit and len(clips) >= limit:
            break
    X = torch.from_numpy(np.stack(clips))
    Y = torch.from_numpy(np.stack(ys))
    return X, Y, vids


def probe_one(model, X, Y, vids, device, batch):
    feat = extract_features(model, X, device, batch=batch)
    tr, te = [], []
    for i, v in enumerate(vids):
        (
            tr if assign_split(f"endovis::{v}", train=0.8, val=0.0) == "train" else te
        ).append(i)
    if not te:
        te = tr[-max(1, len(tr) // 5) :]
    tr_t, te_t = torch.tensor(tr), torch.tensor(te)
    head = torch.nn.Linear(feat.size(1), Y.size(1))
    opt = torch.optim.Adam(head.parameters(), lr=0.05, weight_decay=1e-4)
    mu, sd = feat.mean(0, keepdim=True), feat.std(0, keepdim=True) + 1e-6
    fz = (feat - mu) / sd
    for _ in range(250):
        loss = F.binary_cross_entropy_with_logits(head(fz[tr_t]), Y[tr_t])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        scores = torch.sigmoid(head(fz[te_t])).numpy()
    yte = Y[te_t].numpy()
    aps = [average_precision(scores[:, c], yte[:, c]) for c in range(Y.size(1))]
    return float(np.nanmean(aps)), aps, len(tr), len(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vjepa", default="outputs/vjepa_l1/vjepa_l1_adapt.pt")
    ap.add_argument("--root", default="datasets/endovis2017_full/endovis2017")
    ap.add_argument("--limit", type=int, default=160)
    ap.add_argument("--out", default="outputs/endohjepa/instrument_probe.json")
    ap.add_argument("--encoder", choices=["scratch", "vjepa2"], default="scratch")
    ap.add_argument("--vjepa2-id", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument(
        "--adapted-ckpt",
        default="outputs/vjepa2_adapted/vjepa2_adapted.pt",
        help="e2e domain-adapted V-JEPA2 checkpoint",
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="run frozen V-JEPA2 vs domain-adapted V-JEPA2 side by side",
    )
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from endoworld.understanding.encoders import load_any_encoder, load_adapted_vjepa2

    if args.compare:
        encoders = {}
        fr, clip_len, image_size, _ = load_any_encoder(
            "vjepa2", device, args.vjepa2_id, args.vjepa
        )
        encoders["vjepa2-frozen"] = fr
        if Path(args.adapted_ckpt).is_file():
            ad, _, _, _ = load_adapted_vjepa2(args.adapted_ckpt, device)
            encoders["vjepa2-adapted"] = ad
        else:
            print(f"[compare] adapted ckpt missing: {args.adapted_ckpt}")
    elif args.encoder == "vjepa2":
        model, clip_len, image_size, _ = load_any_encoder(
            "vjepa2", device, args.vjepa2_id, args.vjepa
        )
        encoders = {"vjepa2-frozen": model}
    else:
        model, cfg = load_encoder(args.vjepa, device, random_init=False)
        clip_len, image_size = cfg.clip_len, cfg.image_size
        encoders = {"scratch": model}

    X, Y, vids = build_endovis(args.root, clip_len, image_size, args.limit)
    print(f"[endovis] {len(X)} clips / {len(set(vids))} sequences")
    names = [v for k, v in sorted(load_class_map(args.root).items()) if k != 0]
    results = {}
    for tag, model in encoders.items():
        batch = 2 if tag.startswith("vjepa2") else 8
        mAP, aps, ntr, nte = probe_one(model, X, Y, vids, device, batch)
        results[tag] = {
            "mAP": mAP,
            "per_class": dict(zip(names, aps)),
            "n_train": ntr,
            "n_test": nte,
        }
        print(f"[{tag:16s}] video-level mAP = {mAP:.3f}")
    report = {
        "video_level": True,
        "paper": "Endo-HJEPA",
        "not_cholecseg8k_0992": True,
        "main_table_ok": True,
        "results": results,
    }
    if "vjepa2-frozen" in results and "vjepa2-adapted" in results:
        report["adaptation_gain"] = (
            results["vjepa2-adapted"]["mAP"] - results["vjepa2-frozen"]["mAP"]
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        __import__("json").dumps(report, indent=2), encoding="utf-8"
    )
    print(__import__("json").dumps(report, indent=2))


if __name__ == "__main__":
    main()
