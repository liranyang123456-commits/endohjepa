"""L1 endoscopic domain adaptation: continue JEPA pretrain with specular down-weighting.

Also reports a multi-horizon persistence vs linear predictor baseline on frozen
(or adapted) temporal latents.

    python -m endoworld.understanding.adapt --smoke
    python -m endoworld.understanding.adapt --epochs 8 --max-clips 400 --endo-mask
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from endoworld.data.video_clips import EndoClipDataset, domain_balanced_indices
from endoworld.understanding.endo_mask import token_loss_weights
from endoworld.understanding.l1_regularizers import temporal_smoothness
from endoworld.understanding.vjepa import VJEPA, VJEPAConfig
from endoworld.world.h_jepa import persistence_baseline


def _cli_values(values) -> set[str]:
    """Normalise repeatable comma-separated CLI values."""
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    return {
        item.strip().lower()
        for value in values
        for item in value.split(",")
        if item.strip()
    }


def _sequence_tokens(sequence_id: str) -> set[str]:
    normalised = str(sequence_id).replace("\\", "/").lower()
    return {token for token in normalised.split("/") if token}


def filter_adaptation_clips(
    clips,
    allowed_splits: set[str],
    excluded_datasets: set[str],
    excluded_video_ids: set[str],
    allow_cholect50_held_out: bool,
):
    """Exclude held-out videos and explicitly disallowed data from adaptation."""
    from endoworld.data.cholect50 import CHALLENGE_TEST_VIDS

    challenge_ids = {str(video).lower() for video in CHALLENGE_TEST_VIDS}
    selected = []
    for clip in clips:
        dataset = str(clip.dataset).lower()
        split = str(getattr(clip, "split", "")).lower()
        tokens = _sequence_tokens(clip.sequence_id)
        if allowed_splits and split not in allowed_splits:
            continue
        if dataset in excluded_datasets:
            continue
        if tokens & excluded_video_ids:
            continue
        if (
            not allow_cholect50_held_out
            and dataset == "cholect50"
            and tokens & challenge_ids
        ):
            continue
        selected.append(clip)
    return selected


def _adaptation_audit(
    clips,
    args,
    allowed_splits: set[str],
    excluded_datasets: set[str],
    excluded_video_ids: set[str],
) -> dict:
    clip_ids = sorted(str(clip.sequence_id) for clip in clips)
    digest = hashlib.sha256("\n".join(clip_ids).encode("utf-8")).hexdigest()
    return {
        "manifest": args.manifest,
        "n_clips": len(clips),
        "clip_ids": clip_ids,
        "clip_ids_sha256": digest,
        "allowed_splits": sorted(allowed_splits),
        "excluded_datasets": sorted(excluded_datasets),
        "excluded_video_ids": sorted(excluded_video_ids),
        "allow_cholect50_held_out": bool(args.allow_cholect50_held_out),
    }


def _filter_adaptation_dataset(ds, args) -> dict:
    allowed_splits = _cli_values(args.allow_splits)
    excluded_datasets = _cli_values(args.exclude_datasets)
    excluded_video_ids = _cli_values(args.exclude_video_ids)
    ds.clips = filter_adaptation_clips(
        ds.clips,
        allowed_splits,
        excluded_datasets,
        excluded_video_ids,
        args.allow_cholect50_held_out,
    )
    return _adaptation_audit(
        ds.clips, args, allowed_splits, excluded_datasets, excluded_video_ids
    )


def cosine_lr(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * p))


def collate(batch):
    return torch.stack([b if torch.is_tensor(b) else torch.as_tensor(b) for b in batch])


def _subsample(ds, max_clips, balance, seed=0):
    if max_clips and len(ds.clips) > max_clips:
        if balance:
            idx = domain_balanced_indices(ds.clips, n=max_clips, seed=seed)
            ds.clips = [ds.clips[i] for i in idx]
        else:
            import random

            random.seed(seed)
            ds.clips = random.sample(ds.clips, max_clips)


def multi_horizon_baseline(Z: torch.Tensor, history: int) -> list[dict]:
    """Tiny linear predictor vs persistence on cached (N,T,D) latents."""
    t = Z.size(1)
    horizon = max(1, t - history)
    z_hist, z_fut = Z[:, :history], Z[:, history : history + horizon]
    # closed-form last-frame copy + a learned Linear on flattened history
    persist = persistence_baseline(z_hist, horizon)
    x = z_hist.reshape(Z.size(0), -1)
    y = z_fut.reshape(Z.size(0), -1)
    n = x.size(0)
    n_tr = max(1, int(0.8 * n))
    xt, yt = x[:n_tr], y[:n_tr]
    w = torch.linalg.lstsq(xt, yt).solution
    pred = (x @ w).view_as(z_fut)
    rows = []
    for h in (1, 4, 8, 16):
        if h > horizon:
            continue
        rows.append(
            {
                "horizon": h,
                "cos_linear": F.cosine_similarity(pred[:, :h], z_fut[:, :h], dim=-1)
                .mean()
                .item(),
                "cos_persist": F.cosine_similarity(persist[:, :h], z_fut[:, :h], dim=-1)
                .mean()
                .item(),
                "mse_linear": (pred[:, :h] - z_fut[:, :h]).pow(2).mean().item(),
                "mse_persist": (persist[:, :h] - z_fut[:, :h]).pow(2).mean().item(),
            }
        )
    return rows


def train_vjepa2_l1(args, device):
    """Domain-adapt a predictor on official V-JEPA2 dense tokens (encoder frozen by default)."""
    from endoworld.understanding.vjepa2_hf import VJEPA2Encoder
    from endoworld.world.h_jepa import EndoHJEPA, HJEPAConfig
    from endoworld.world.train import cache_latents, collate_meta

    enc = VJEPA2Encoder(args.vjepa2_id, device=device, unfreeze_last=args.unfreeze_last)
    clip_len, image_size, dim = 16, enc.image_size, enc.embed_dim
    ds = EndoClipDataset(
        args.manifest,
        clip_len=clip_len,
        stride=args.stride,
        image_size=image_size,
        exclude=["EndoVis2019_ROBUST-MIS"],
        split=None if args.smoke else args.split,
        return_meta=True,
    )
    audit = _filter_adaptation_dataset(ds, args)
    if args.smoke:
        ds.clips = ds.clips[: max(args.smoke_clips, 4)]
    else:
        _subsample(ds, args.max_clips, args.balance_domains)
    print(f"[data] {len(ds)} clips  encoder=vjepa2  dim={dim}")
    dl = DataLoader(
        ds,
        batch_size=max(1, args.batch_size // 4),
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_meta,
        drop_last=True,
    )
    print("[cache] V-JEPA2 dense tokens ...")
    Z, D = cache_latents(enc, dl, device, dense=True)
    t = Z.size(1)
    history = min(4, t - 1)
    horizon = min(4, t - history)
    wcfg = HJEPAConfig(
        latent_dim=dim,
        hidden_dim=min(512, dim),
        n_heads=8,
        n_layers=4,
        history=history,
        horizon=horizon,
        spatial_keep=args.spatial_keep,
        ablation="l1",
    )
    model = EndoHJEPA(wcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    os.makedirs(args.out, exist_ok=True)
    Z, D = Z.to(device), D.to(device)
    bs = 4
    for epoch in range(args.epochs):
        model.train()
        idx = torch.randperm(Z.size(0), device=device)
        run = 0.0
        n = 0
        for i in range(0, Z.size(0), bs):
            sl = idx[i : i + bs]
            if sl.numel() < 2:
                continue
            out = model.losses_dense(Z[sl], D[sl], history, horizon)
            loss = out["total"]
            if args.smooth_reg > 0:
                loss = loss + args.smooth_reg * temporal_smoothness(Z[sl])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item() * sl.numel()
            n += sl.numel()
        print(f"[epoch {epoch}] vjepa2-l1 loss={run / max(n, 1):.4f}")
    torch.save(
        {
            "model": model.state_dict(),
            "wcfg": wcfg.__dict__,
            "history": history,
            "horizon": horizon,
            "kind": "hjepa",
            "ablation": "l1",
            "dense": True,
            "embed_dim": dim,
            "adaptation_audit": audit,
        },
        os.path.join(args.out, "vjepa2_l1.pt"),
    )
    torch.save(
        {"Z": Z.cpu(), "D": D.cpu(), "dense": True},
        os.path.join(args.out, "latents_cache.pt"),
    )
    print(f"[ckpt] {args.out}")
    if args.stir_reg > 0:
        _dump_stir_metric(enc, args, device)


def train_vjepa2_e2e(args, device):
    """End-to-end domain adaptation: fine-tune the last K V-JEPA2 blocks through the
    world-model loss so the encoder produces more predictable endoscopic tokens.

    Unlike train_vjepa2_l1 (frozen encoder, cached tokens), this backpropagates into
    the unfrozen encoder blocks. Batch size is forced to 1 (ViT-L activation memory).
    """
    from endoworld.understanding.vjepa2_hf import VJEPA2Encoder
    from endoworld.world.h_jepa import EndoHJEPA, HJEPAConfig
    from endoworld.world.train import collate_meta

    if args.unfreeze_last <= 0:
        raise ValueError("--e2e requires --unfreeze-last K > 0")
    enc = VJEPA2Encoder(args.vjepa2_id, device=device, unfreeze_last=args.unfreeze_last)
    clip_len, image_size, dim = 16, enc.image_size, enc.embed_dim
    ds = EndoClipDataset(
        args.manifest,
        clip_len=clip_len,
        stride=args.stride,
        image_size=image_size,
        exclude=["EndoVis2019_ROBUST-MIS"],
        split=None if args.smoke else args.split,
        return_meta=True,
    )
    audit = _filter_adaptation_dataset(ds, args)
    if args.smoke:
        ds.clips = ds.clips[: max(args.smoke_clips, 4)]
    else:
        _subsample(ds, args.max_clips, args.balance_domains)
    print(f"[data] {len(ds)} clips  e2e unfreeze_last={args.unfreeze_last}  dim={dim}")
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_meta,
        drop_last=True,
    )
    history, horizon = 4, 4
    wcfg = HJEPAConfig(
        latent_dim=dim,
        hidden_dim=min(512, dim),
        n_heads=8,
        n_layers=2,
        history=history,
        horizon=horizon,
        spatial_keep=args.spatial_keep,
        ablation="l1",
        predictor="spacetime",
    )
    model = EndoHJEPA(wcfg).to(device)
    params = list(model.parameters()) + enc.trainable_parameters()
    n_train = sum(p.numel() for p in enc.trainable_parameters()) / 1e6
    print(
        f"[e2e] encoder trainable {n_train:.1f}M + predictor {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
    )
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    os.makedirs(args.out, exist_ok=True)
    enc.train()
    for epoch in range(args.epochs):
        model.train()
        run, n = 0.0, 0
        for clip, dom in dl:
            clip = clip.to(device).float()
            z = enc.encode_dense(clip)
            loss = model.losses_dense(z, dom.to(device), history, horizon)["total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            run += loss.item()
            n += 1
        print(f"[epoch {epoch}] e2e loss={run / max(n, 1):.4f}")
    torch.save(
        {
            "encoder": enc.model.state_dict(),
            "model": model.state_dict(),
            "wcfg": wcfg.__dict__,
            "unfreeze_last": args.unfreeze_last,
            "vjepa2_id": args.vjepa2_id,
            "kind": "hjepa",
            "ablation": "l1",
            "dense": True,
            "embed_dim": dim,
            "history": history,
            "horizon": horizon,
            "adaptation_audit": audit,
        },
        os.path.join(args.out, "vjepa2_adapted.pt"),
    )
    print(f"[ckpt] {args.out}/vjepa2_adapted.pt")


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    if args.encoder == "vjepa2" and not args.smoke:
        if args.e2e:
            train_vjepa2_e2e(args, device)
        else:
            train_vjepa2_l1(args, device)
        return

    if args.smoke:
        cfg = VJEPAConfig(
            image_size=64,
            clip_len=8,
            embed_dim=128,
            depth=2,
            num_heads=4,
            predictor_dim=96,
            predictor_depth=2,
        )
    else:
        cfg = VJEPAConfig(
            image_size=args.image_size,
            clip_len=args.clip_len,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.heads,
            mask_ratio=args.mask_ratio,
        )

    split = None if args.smoke else args.split
    ds = EndoClipDataset(
        args.manifest,
        clip_len=cfg.clip_len,
        stride=args.stride,
        image_size=cfg.image_size,
        exclude=["EndoVis2019_ROBUST-MIS"],
        split=split,
    )
    if args.smoke:
        if len(ds) == 0:
            raise RuntimeError("no clips; rebuild manifests first")
        ds.clips = ds.clips[: max(args.smoke_clips, 4)]
    else:
        _subsample(ds, args.max_clips, args.balance_domains)
    print(f"[data] {len(ds)} clips  split={split or 'all'}")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        drop_last=True,
    )

    model = VJEPA(cfg).to(device)
    if (not args.smoke) and args.init and os.path.isfile(args.init):
        blob = torch.load(args.init, map_location=device, weights_only=False)
        try:
            model.load_state_dict(blob["model"], strict=False)
            print(f"[init] loaded {args.init}")
        except RuntimeError as e:
            print(f"[init] skip incompatible {args.init}: {e.__class__.__name__}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"[model] VJEPA L1-adapt {n_params:.1f}M  tokens={cfg.n_tokens}  endo_mask={args.endo_mask}"
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.05,
    )
    total_steps = args.epochs * max(len(dl), 1)
    warmup = max(int(0.05 * total_steps), 1)
    os.makedirs(args.out, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        run = 0.0
        for clip in dl:
            clip = clip.to(device, non_blocking=True).float()
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total_steps, args.lr, warmup)
            weights = None
            if args.endo_mask:
                weights = token_loss_weights(clip, cfg.tubelet_size, cfg.patch_size).to(
                    device
                )
            loss = model(clip, token_weights=weights)
            if args.smooth_reg > 0:
                with torch.no_grad():
                    z = model.encode_dense(clip)
                loss = loss + args.smooth_reg * temporal_smoothness(z)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.update_target()
            run += loss.item()
            step += 1
        avg = run / max(len(dl), 1)
        if args.instrument_mask:
            extra = _endovis_mask_steps(model, opt, device, cfg, args)
            print(
                f"[epoch {epoch}] avg_loss={avg:.4f} time={time.time() - t0:.1f}s  endovis_steps={extra}"
            )
        else:
            print(f"[epoch {epoch}] avg_loss={avg:.4f} time={time.time() - t0:.1f}s")

    ckpt = os.path.join(args.out, "vjepa_l1_adapt.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt)
    print(f"[ckpt] {ckpt}")

    model.eval()
    Z = []
    with torch.no_grad():
        for clip in dl:
            clip = clip.to(device).float()
            Z.append(model.encode_temporal(clip).cpu())
            if args.smoke and len(Z) >= 2:
                break
    if Z:
        Z = torch.cat(Z)
        hist = min(4, Z.size(1) - 1)
        rows = multi_horizon_baseline(Z, hist)
        bench = os.path.join(args.out, "l1_horizon_baseline.json")
        with open(bench, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print("[horizon baseline]")
        print(json.dumps(rows, indent=2))
    if args.smoke:
        print("[smoke] L1 domain adapt OK")


def _dump_stir_metric(enc, args, device):
    try:
        from endoworld.data.stir_tracks import find_stir_sequences
        from endoworld.eval.stir_experiment import evaluate

        seqs = find_stir_sequences("datasets/STIR")
        image_size = getattr(enc, "image_size", args.image_size)
        report = evaluate(enc, seqs, image_size, device, limit=16)
        path = os.path.join(args.out, "stir_metric.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[stir] mean_chamfer={report.get('mean_chamfer')} -> {path}")
    except Exception as e:
        print(f"[stir] skip: {e}")


def _endovis_mask_steps(model, opt, device, cfg, args) -> int:
    from endoworld.data.endovis_masks import iter_endovis_clips

    n = 0
    root = args.endovis_root
    for _, clip, inst in iter_endovis_clips(
        root, "train", cfg.clip_len, cfg.image_size, limit=4
    ):
        clip = clip.unsqueeze(0).to(device).float()
        inst = inst.unsqueeze(0).to(device)
        weights = token_loss_weights(
            clip,
            cfg.tubelet_size,
            cfg.patch_size,
            instrument_mask=inst,
            instrument_boost=args.instrument_boost,
        )
        loss = model(clip, token_weights=weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.update_target()
        n += 1
    return n


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/sequences.csv")
    ap.add_argument("--out", default="outputs/vjepa_l1")
    ap.add_argument("--init", default="outputs/vjepa/vjepa_epoch8.pt")
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--allow-splits",
        action="append",
        default=["train"],
        help="repeatable/comma-separated manifest splits allowed for adaptation",
    )
    ap.add_argument(
        "--exclude-datasets",
        action="append",
        default=[],
        help="repeatable/comma-separated dataset names excluded from adaptation",
    )
    ap.add_argument(
        "--exclude-video-ids",
        action="append",
        default=[],
        help="repeatable/comma-separated video IDs excluded from adaptation",
    )
    ap.add_argument(
        "--allow-cholect50-held-out",
        action="store_true",
        help="explicitly permit official CholecT50 challenge-test videos",
    )
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-clips", type=int, default=400)
    ap.add_argument("--endo-mask", action="store_true", default=True)
    ap.add_argument("--no-endo-mask", action="store_false", dest="endo_mask")
    ap.add_argument("--balance-domains", action="store_true", default=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-clips", type=int, default=8)
    ap.add_argument("--encoder", choices=["scratch", "vjepa2"], default="scratch")
    ap.add_argument("--vjepa2-id", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--unfreeze-last", type=int, default=0)
    ap.add_argument(
        "--e2e",
        action="store_true",
        help="end-to-end: fine-tune the last --unfreeze-last encoder blocks",
    )
    ap.add_argument("--spatial-keep", type=int, default=16)
    ap.add_argument(
        "--smooth-reg",
        type=float,
        default=0.05,
        help="temporal smoothness on dense tokens",
    )
    ap.add_argument(
        "--stir-reg",
        type=float,
        default=0.0,
        help=">0: dump STIR endpoint chamfer after V-JEPA2 adapt",
    )
    ap.add_argument(
        "--instrument-mask",
        action="store_true",
        help="extra EndoVis mask-weighted JEPA steps (scratch encoder)",
    )
    ap.add_argument("--instrument-boost", type=float, default=1.0)
    ap.add_argument("--endovis-root", default="datasets/endovis2017_full/endovis2017")
    return ap


if __name__ == "__main__":
    train(build_argparser().parse_args())
