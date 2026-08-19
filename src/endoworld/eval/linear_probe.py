"""Linear-probe evaluation of the V-JEPA representation on CholecSeg8k.

Task: multi-label presence of anatomy/instrument classes (from labels_13cls masks).
Protocol: freeze the encoder, extract one pooled embedding per clip, then fit a single
linear layer with BCE. We compare the pretrained encoder against a random-init encoder
of the same architecture -- a higher mean-AP for the pretrained one shows the
self-supervised representation captures endoscopic semantics.

    python -m endoworld.eval.linear_probe --vjepa outputs/vjepa/vjepa_epoch8.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from endoworld.understanding.vjepa import VJEPA, VJEPAConfig
from endoworld.captioning.build_caption_pairs import CHOLECSEG8K_CLASSES

# classes worth probing (drop background id 0)
PROBE_IDS = [i for i in CHOLECSEG8K_CLASSES if i != 0]


def build_clips(
    images_dir, labels_dir, clip_len, stride, image_size, limit, max_clip_stride
):
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    files = sorted(f for f in os.listdir(images_dir) if f.endswith(".png"))
    # group by video prefix (video01_frame_XXX_endo.png)
    from collections import defaultdict

    groups = defaultdict(list)
    for f in files:
        vid = f.split("_frame_")[0]
        groups[vid].append(f)

    span = (clip_len - 1) * max_clip_stride + 1
    clips = []
    vids = []
    for vid, fl in groups.items():
        fl = sorted(fl)
        for start in range(0, len(fl) - span + 1, span):
            idxs = [start + i * max_clip_stride for i in range(clip_len)]
            clips.append([fl[i] for i in idxs])
            vids.append(vid)
    if limit:
        rng = np.random.default_rng(0)
        pick = rng.permutation(len(clips))[:limit]
        clips = [clips[i] for i in pick]
        vids = [vids[i] for i in pick]

    X_frames, Y = [], []
    for names in clips:
        frames = []
        for n in names:
            img = (
                Image.open(os.path.join(images_dir, n))
                .convert("RGB")
                .resize((image_size, image_size))
            )
            frames.append(np.asarray(img, np.float32) / 255.0)
        # label from center frame mask
        center = names[len(names) // 2]
        lab = np.asarray(Image.open(os.path.join(labels_dir, center)))
        if lab.ndim == 3:
            lab = lab[..., 0]
        present = set(int(v) for v in np.unique(lab))
        y = np.array([1.0 if i in present else 0.0 for i in PROBE_IDS], np.float32)
        X_frames.append(np.stack(frames))  # (T,H,W,C)
        Y.append(y)
    X = np.stack(X_frames).transpose(0, 1, 4, 2, 3)  # (N,T,C,H,W)
    return torch.from_numpy(X), torch.from_numpy(np.stack(Y)), vids


@torch.no_grad()
def extract_features(model, X, device, batch=16):
    feats = []
    for i in range(0, len(X), batch):
        clip = X[i : i + batch].to(device).float()
        feats.append(model.encode(clip).cpu())
    return torch.cat(feats)


def average_precision(scores, labels):
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    prec = tp / (np.arange(len(labels)) + 1)
    n_pos = labels.sum()
    return float((prec * labels).sum() / n_pos) if n_pos > 0 else float("nan")


def _video_level_indices(vids, train_frac=0.8, seed=0):
    """All clips from one video stay on the same side of the split."""
    from endoworld.data.splits import assign_split

    tr, te = [], []
    for i, vid in enumerate(vids):
        split = assign_split(f"probe::{vid}", seed=seed, train=train_frac, val=0.0)
        (tr if split == "train" else te).append(i)
    if not tr or not te:
        n = len(vids)
        n_tr = max(1, int(train_frac * n))
        perm = list(range(n))
        return perm[:n_tr], perm[n_tr:] or perm[-1:]
    return tr, te


def fit_linear_probe(feat, Y, epochs=300, lr=0.05, vids=None, video_level=True):
    d = feat.size(1)
    head = torch.nn.Linear(d, Y.size(1))
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    fmean, fstd = feat.mean(0, keepdim=True), feat.std(0, keepdim=True) + 1e-6
    fz = (feat - fmean) / fstd
    n = len(fz)
    if video_level and vids is not None:
        tr, te = _video_level_indices(vids)
        tr, te = torch.tensor(tr), torch.tensor(te)
    else:
        idx = torch.randperm(n)
        n_tr = int(0.8 * n)
        tr, te = idx[:n_tr], idx[n_tr:]
    for _ in range(epochs):
        logit = head(fz[tr])
        loss = F.binary_cross_entropy_with_logits(logit, Y[tr])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        scores = torch.sigmoid(head(fz[te])).numpy()
    yte = Y[te].numpy()
    aps = [average_precision(scores[:, c], yte[:, c]) for c in range(Y.size(1))]
    return np.nanmean(aps), aps


def load_encoder(ckpt, device, random_init=False):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = VJEPAConfig(**ck["cfg"])
    model = VJEPA(cfg).to(device).eval()
    if not random_init:
        model.load_state_dict(ck["model"])
    return model, cfg


def run_probe(build_kwargs, encoders, args, device, batch=16):
    print(
        f"[probe] building clips (clip_len={build_kwargs['clip_len']}, "
        f"img={build_kwargs['image_size']}, stride={build_kwargs['stride']}) ..."
    )
    X, Y, vids = build_clips(
        args.images,
        args.labels,
        build_kwargs["clip_len"],
        build_kwargs["stride"],
        build_kwargs["image_size"],
        args.limit,
        build_kwargs["stride"],
    )
    print(
        f"[probe] {len(X)} clips / {len(set(vids))} videos; class positives: "
        f"{ {CHOLECSEG8K_CLASSES[PROBE_IDS[c]]: int(Y[:, c].sum()) for c in range(Y.size(1))} }"
    )
    print("[probe] video-level split (do not cite clip-leaky 0.992 mAP in Endo-HJEPA)")
    results = {}
    for tag, model in encoders.items():
        feat = extract_features(model, X, device, batch=batch)
        mAP, _ = fit_linear_probe(feat, Y, vids=vids, video_level=args.video_level)
        results[tag] = mAP
        print(f"[{tag:16s}] linear-probe mean AP = {mAP:.3f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--encoder", choices=["scratch", "vjepa2", "all"], default="scratch"
    )
    ap.add_argument("--vjepa", default="outputs/vjepa/vjepa_epoch8.pt")
    ap.add_argument("--vjepa2-id", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--images", default="datasets/CholecSeg8k/images")
    ap.add_argument("--labels", default="datasets/CholecSeg8k/labels_13cls")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--clip-stride", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--vjepa2-batch", type=int, default=2)
    ap.add_argument("--video-level", action="store_true", default=True)
    ap.add_argument("--clip-level", action="store_false", dest="video_level")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.encoder in ("scratch", "all"):
        pre, cfg = load_encoder(args.vjepa, device, random_init=False)
        rnd, _ = load_encoder(args.vjepa, device, random_init=True)
        run_probe(
            {
                "clip_len": cfg.clip_len,
                "image_size": cfg.image_size,
                "stride": args.clip_stride,
            },
            {"scratch-pretrained": pre, "scratch-random": rnd},
            args,
            device,
            batch=args.batch,
        )

    if args.encoder in ("vjepa2", "all"):
        from endoworld.understanding.vjepa2_hf import VJEPA2Encoder

        enc = VJEPA2Encoder(args.vjepa2_id, device=device)
        run_probe(
            {"clip_len": 64, "image_size": enc.image_size, "stride": 1},
            {"vjepa2-official": enc},
            args,
            device,
            batch=args.vjepa2_batch,
        )


if __name__ == "__main__":
    main()
