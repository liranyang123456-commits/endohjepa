"""Train Endo-HJEPA on encoder latents (pooled or dense spatiotemporal tokens).

python -m endoworld.world.train --smoke
python -m endoworld.world.train --ablation l1 --dense --split-aware
python -m endoworld.world.train --ablation full --encoder scratch --epochs 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from endoworld.data.video_clips import EndoClipDataset, domain_balanced_indices
from endoworld.world.baselines import GRUDynamics
from endoworld.world.h_jepa import EndoHJEPA, HJEPAConfig, persistence_baseline


def _sha256_file(path: str) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collate_meta(batch):
    clips, domains = zip(*batch)
    return torch.stack(clips), torch.tensor(domains, dtype=torch.long)


def load_encoder(args, device):
    if args.encoder == "vjepa2":
        adapted = getattr(args, "vjepa2_adapted", "")
        if adapted and os.path.isfile(adapted):
            from endoworld.understanding.encoders import load_adapted_vjepa2

            enc, _, image_size, embed_dim = load_adapted_vjepa2(adapted, device)
            print(f"[encoder] domain-adapted V-JEPA2 <- {adapted}")
            return enc, 64, image_size, embed_dim
        from endoworld.understanding.vjepa2_hf import VJEPA2Encoder

        enc = VJEPA2Encoder(args.vjepa2_id, device=device)
        return enc, 64, enc.image_size, enc.embed_dim
    from endoworld.understanding.vjepa import VJEPA, VJEPAConfig

    if args.vjepa and os.path.isfile(args.vjepa):
        ck = torch.load(args.vjepa, map_location=device, weights_only=False)
        cfg = VJEPAConfig(**ck["cfg"])
        model = VJEPA(cfg).to(device).eval()
        model.load_state_dict(ck["model"])
        for p in model.parameters():
            p.requires_grad_(False)
        return model, cfg.clip_len, cfg.image_size, cfg.embed_dim
    print(f"[encoder] {args.vjepa} missing; using random compact VJEPA")
    cfg = VJEPAConfig(
        image_size=96,
        clip_len=16,
        embed_dim=256,
        depth=4,
        num_heads=4,
        predictor_dim=128,
        predictor_depth=2,
    )
    model = VJEPA(cfg).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg.clip_len, cfg.image_size, cfg.embed_dim


def _take_clips(ds, args):
    if args.smoke:
        if len(ds) == 0:
            raise RuntimeError("no clips in manifest; rebuild manifests first")
        ds.clips = ds.clips[: max(args.smoke_clips, 4)]
        return
    if args.max_clips and len(ds.clips) > args.max_clips:
        idx = domain_balanced_indices(ds.clips, n=args.max_clips, seed=0)
        ds.clips = [ds.clips[i] for i in idx]


def cache_latents(enc, dl, device, dense: bool):
    Z, Doms = [], []
    t0 = time.time()
    with torch.no_grad():
        for bi, (clip, domain) in enumerate(dl):
            clip = clip.to(device).float()
            if dense:
                Z.append(enc.encode_dense(clip).cpu())
            else:
                Z.append(enc.encode_temporal(clip).cpu())
            Doms.append(domain)
            if (bi + 1) % 10 == 0:
                print(
                    f"  encoded {sum(z.shape[0] for z in Z)} ({time.time() - t0:.0f}s)"
                )
    return torch.cat(Z), torch.cat(Doms)


def train(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    reuse = bool(args.latents and os.path.isfile(args.latents) and not args.smoke)
    if args.smoke:
        from endoworld.understanding.vjepa import VJEPA, VJEPAConfig

        vcfg = VJEPAConfig(
            image_size=64,
            clip_len=8,
            embed_dim=128,
            depth=2,
            num_heads=4,
            predictor_dim=96,
            predictor_depth=2,
        )
        enc = VJEPA(vcfg).to(device).eval()
        for p in enc.parameters():
            p.requires_grad_(False)
        clip_len, image_size, embed_dim = vcfg.clip_len, vcfg.image_size, vcfg.embed_dim
    elif reuse:
        enc = None
        clip_len, image_size, embed_dim = 16, 256, 1024
        print(f"[encoder] skipped (reusing {args.latents})")
    else:
        enc, clip_len, image_size, embed_dim = load_encoder(args, device)

    if args.clip_len:
        clip_len = (
            args.clip_len
        )  # override to control cache memory (T = clip_len / tubelet)

    split_tr = None if (args.smoke or not args.split_aware) else "train"
    split_va = None if (args.smoke or not args.split_aware) else "val"
    if reuse:
        split_va = None

    pack = None
    if reuse:
        pack = torch.load(args.latents, map_location="cpu", weights_only=False)
        Z, Doms = pack["Z"], pack["D"]
        embed_dim = int(Z.size(-1))
        print(f"[cache] loaded {args.latents} {tuple(Z.shape)}")
        print(
            f"[data] train clips={Z.size(0)} (from cache) dense={args.dense} ablation={args.ablation}"
        )
    else:
        ds = EndoClipDataset(
            args.manifest,
            clip_len=clip_len,
            stride=args.stride,
            image_size=image_size,
            exclude=["EndoVis2019_ROBUST-MIS"],
            return_meta=True,
            split=split_tr,
            include_domains=args.domains.split(",") if args.domains else None,
        )
        _take_clips(ds, args)
        print(
            f"[data] train clips={len(ds)} split={split_tr or 'all'} dense={args.dense} ablation={args.ablation}"
        )
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=collate_meta,
            drop_last=True,
        )
        print("[cache] encoding clips ...")
        Z, Doms = cache_latents(enc, dl, device, args.dense)
        print(f"[cache] latents {tuple(Z.shape)} domains {tuple(Doms.shape)}")

    # pooled-only run requested but the cache is dense: pool spatial tokens once
    if not args.dense and Z.dim() == 4:
        Z = Z.mean(dim=2)
        print(f"[cache] pooled to {tuple(Z.shape)} for non-dense run")

    t_steps = Z.size(1)
    history = min(args.history, t_steps - 1)
    horizon = min(args.horizon, t_steps - history)
    if horizon < 1:
        raise RuntimeError(f"need history+horizon < {t_steps}, got {history}+{horizon}")

    Z_val = D_val = None
    val_is_videolevel = False
    if reuse and pack is not None and pack.get("Z_val") is not None:
        Z_val, D_val = pack["Z_val"], pack["D_val"]
        val_is_videolevel = True
        print(f"[data] val clips={Z_val.size(0)} (video-level split, cached)")
    elif split_va:
        ds_va = EndoClipDataset(
            args.manifest,
            clip_len=clip_len,
            stride=args.stride,
            image_size=image_size,
            exclude=["EndoVis2019_ROBUST-MIS"],
            return_meta=True,
            split=split_va,
            include_domains=args.domains.split(",") if args.domains else None,
        )
        if args.max_clips:
            n_va = max(8, args.max_clips // 8)
            idx = domain_balanced_indices(
                ds_va.clips, n=min(n_va, len(ds_va.clips)), seed=1
            )
            ds_va.clips = [ds_va.clips[i] for i in idx]
        if len(ds_va) >= 2:
            dl_va = DataLoader(
                ds_va,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                collate_fn=collate_meta,
            )
            Z_val, D_val = cache_latents(enc, dl_va, device, args.dense)
            val_is_videolevel = True
            print(f"[data] val clips={Z_val.size(0)} (video-level split)")

    n = Z.size(0)
    if Z_val is None:
        n_val = max(1, int(0.15 * n))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        Z_tr, Z_val = Z[tr_idx], Z[val_idx]
        D_tr, D_val = Doms[tr_idx], Doms[val_idx]
        print("[warn] no val split in manifest; falling back to random clip split")
    else:
        Z_tr, D_tr = Z, Doms

    # Keep the (potentially large) latent cache on CPU; move only each batch to the
    # GPU. This is what makes >1000-clip dense caches feasible.
    Z_tr, D_tr = Z_tr.cpu(), D_tr.cpu()
    Z_val, D_val = Z_val.cpu(), D_val.cpu()

    # scale-curve: subsample the cached training latents to N clips
    if args.train_subset and Z_tr.size(0) > args.train_subset:
        g = torch.Generator().manual_seed(0)
        sub = torch.randperm(Z_tr.size(0), generator=g)[: args.train_subset]
        Z_tr, D_tr = Z_tr[sub], D_tr[sub]
        print(f"[scale] training on subset {Z_tr.size(0)} / {Z.size(0)} clips")

    # zero-shot cross-domain: restrict training to source domains, eval on all
    if args.train_domains:
        from endoworld.data.domains import DOMAIN_IDS

        keep = {
            DOMAIN_IDS[d.strip()] for d in args.train_domains.split(",") if d.strip()
        }
        m = torch.tensor([int(x) in keep for x in D_tr])
        Z_tr, D_tr = Z_tr[m], D_tr[m]
        print(
            f"[transfer] train restricted to {args.train_domains}: {Z_tr.size(0)} clips; "
            f"val covers all domains (zero-shot on held-out domains)"
        )

    if args.cache_only:
        os.makedirs(args.out, exist_ok=True)
        cache_pack = {"Z": Z_tr.cpu(), "D": D_tr.cpu(), "dense": args.dense}
        if val_is_videolevel and Z_val is not None:
            cache_pack["Z_val"] = Z_val.cpu()
            cache_pack["D_val"] = D_val.cpu()
        torch.save(cache_pack, os.path.join(args.out, "latents_cache.pt"))
        print(
            f"[cache-only] saved {tuple(Z_tr.shape)} + val {tuple(Z_val.shape)} -> {args.out}"
        )
        return

    wcfg = HJEPAConfig(
        latent_dim=embed_dim,
        hidden_dim=args.hidden,
        n_heads=args.heads,
        n_layers=args.layers,
        history=history,
        horizon=horizon,
        n_actions=args.n_actions,
        spatial_keep=args.spatial_keep,
        ablation=args.ablation if args.ablation != "gru" else "l1",
        predictor=args.predictor,
        vicreg=args.vicreg,
        unc_weight=not args.no_unc_weight,
        residual=not args.no_residual,
        l1_causal=args.l1_causal,
        query_mask=args.query_mask,
    )

    if args.ablation == "gru":
        model = GRUDynamics(embed_dim, args.hidden, horizon).to(device)
        kind = "gru"
    elif args.ablation == "mamba":
        from endoworld.world.baselines import MambaDynamics

        model = MambaDynamics(embed_dim, args.hidden, horizon).to(device)
        kind = "mamba"
    else:
        model = EndoHJEPA(wcfg).to(device)
        kind = "hjepa"
    print(
        f"[model] {kind}/{args.ablation} {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    os.makedirs(args.out, exist_ok=True)
    bs = args.batch_size

    for epoch in range(args.epochs):
        model.train()
        idx = torch.randperm(Z_tr.size(0))
        run = 0.0
        n_seen = 0
        for i in range(0, Z_tr.size(0), bs):
            sl = idx[i : i + bs]
            if sl.numel() < 2:
                continue
            z = Z_tr[sl].to(device, non_blocking=True)
            d = D_tr[sl].to(device, non_blocking=True)
            if kind in ("gru", "mamba"):
                z_pool = z.mean(dim=2) if z.dim() == 4 else z
                z_hist, z_future = (
                    z_pool[:, :history],
                    z_pool[:, history : history + horizon],
                )
                pred = model(z_hist, d)
                loss = F.smooth_l1_loss(pred, z_future)
            elif args.dense:
                loss = model.losses_dense(z, d, history, horizon)["total"]
            else:
                loss = model.losses(z, d, history, horizon)["total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item() * z.size(0)
            n_seen += z.size(0)
        if (epoch + 1) % max(args.epochs // 10, 1) == 0 or epoch == 0:
            print(f"[epoch {epoch}] train_loss={run / max(n_seen, 1):.4f}")

    model.eval()
    training_parameters = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "history": history,
        "horizon": horizon,
        "query_mask": args.query_mask,
    }
    metric_definition = {
        "aggregation": "mean_over_clips_steps_1_to_H_and_latent_dimensions",
        "cosine": "mean cosine similarity over clips and rollout steps 1..H",
        "mse": "mean squared error over clips, rollout steps 1..H, and latent dimensions",
    }
    cache_sha256 = _sha256_file(args.latents) if reuse else None
    manifest_sha256 = _sha256_file(args.manifest)
    metrics = {
        "seed": args.seed,
        "training_parameters": training_parameters,
        "cache_sha256": cache_sha256,
        "manifest_sha256": manifest_sha256,
        "metric_definition": metric_definition,
    }
    with torch.no_grad():
        Z_val = Z_val.to(device)
        D_val = D_val.to(device)
        is_dense = Z_val.dim() == 4
        z_val = Z_val.mean(dim=2) if is_dense else Z_val
        z_hist, z_future = z_val[:, :history], z_val[:, history : history + horizon]
        persist = persistence_baseline(z_hist, horizon)
        if kind in ("gru", "mamba"):
            pred = model(z_hist)
        else:
            pred = model.forward_l1(z_hist, D_val)
        metrics["cos_model"] = F.cosine_similarity(pred, z_future, dim=-1).mean().item()
        metrics["cos_persist"] = (
            F.cosine_similarity(persist, z_future, dim=-1).mean().item()
        )
        metrics["mse_model"] = (pred - z_future).pow(2).mean().item()
        metrics["mse_persist"] = (persist - z_future).pow(2).mean().item()
        # dense path: evaluate the L1 dense predictor directly and pool its output.
        # only meaningful when the model was actually trained in dense mode.
        if is_dense and args.dense and kind == "hjepa":
            zh_d = Z_val[:, :history]
            zf_d = Z_val[:, history : history + horizon].mean(dim=2)
            pred_d = model.forward_l1_dense(zh_d, D_val).mean(dim=2)
            metrics["cos_model_dense"] = (
                F.cosine_similarity(pred_d, zf_d, dim=-1).mean().item()
            )
            metrics["mse_model_dense"] = (pred_d - zf_d).pow(2).mean().item()
        if kind == "hjepa" and args.ablation == "full":
            z_goal = z_future[:, -1]
            plan_a, plan_e = model.plan(
                z_hist, z_goal, D_val, n_samples=8, steps=horizon
            )
            metrics["plan_energy"] = plan_e.mean().item()
            metrics["plan_actions_shape"] = list(plan_a.shape)
    print(
        f"[rollout|val] cos  model={metrics['cos_model']:.3f}  persistence={metrics['cos_persist']:.3f}"
    )
    if "cos_model_dense" in metrics:
        print(f"[rollout|val] cos  spacetime-dense={metrics['cos_model_dense']:.3f}")
    print(
        f"[rollout|val] mse  model={metrics['mse_model']:.4f}  persistence={metrics['mse_persist']:.4f}"
    )

    ckpt = os.path.join(args.out, "endohjepa.pt")
    torch.save(
        {
            "kind": kind,
            "ablation": args.ablation,
            "dense": args.dense,
            "model": model.state_dict(),
            "wcfg": wcfg.__dict__,
            "history": history,
            "horizon": horizon,
            "embed_dim": embed_dim,
            "seed": args.seed,
            "training_parameters": training_parameters,
            "cache_sha256": cache_sha256,
            "manifest_sha256": manifest_sha256,
            "metric_definition": metric_definition,
        },
        ckpt,
    )

    # Save a compact cache by default: pooled latents are enough for forecast /
    # planning / benchmark eval and are ~256x smaller than dense. Dense saving is
    # opt-in (--save-dense-cache) for reuse in further dense training.
    def _maybe_pool(x):
        return x if (args.save_dense_cache or x.dim() != 4) else x.mean(dim=2)

    cache_pack = {
        "Z": _maybe_pool(Z).cpu(),
        "D": Doms.cpu(),
        "dense": bool(args.dense and args.save_dense_cache),
    }
    if val_is_videolevel and Z_val is not None:
        cache_pack["Z_val"] = _maybe_pool(Z_val).cpu()
        cache_pack["D_val"] = D_val.cpu()
    # dataset names so later evals can do per-dataset decomposition without re-encoding
    if val_is_videolevel and split_va:
        try:
            ds_names = EndoClipDataset(
                args.manifest,
                clip_len=clip_len,
                stride=args.stride,
                image_size=image_size,
                exclude=["EndoVis2019_ROBUST-MIS"],
                return_meta=True,
                split="val",
                include_domains=args.domains.split(",") if args.domains else None,
            )
            n_va = Z_val.size(0)
            idx = domain_balanced_indices(
                ds_names.clips, n=min(n_va, len(ds_names.clips)), seed=1
            )
            cache_pack["datasets_val"] = [ds_names.clips[i].dataset for i in idx[:n_va]]
        except Exception as e:
            print(f"[cache] dataset names not saved: {e}")
    torch.save(cache_pack, os.path.join(args.out, "latents_cache.pt"))
    with open(os.path.join(args.out, "val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[ckpt] {ckpt}")
    if args.smoke:
        print("[smoke] Endo-HJEPA OK")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["scratch", "vjepa2"], default="scratch")
    ap.add_argument("--vjepa", default="outputs/vjepa/vjepa_epoch8.pt")
    ap.add_argument("--vjepa2-id", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument(
        "--vjepa2-adapted",
        default="",
        help="path to e2e domain-adapted V-JEPA2 ckpt (used for caching/training)",
    )
    ap.add_argument("--manifest", default="manifests/sequences.csv")
    ap.add_argument("--out", default="outputs/endohjepa")
    ap.add_argument(
        "--ablation", choices=["l1", "l1l2", "full", "gru", "mamba"], default="full"
    )
    ap.add_argument(
        "--dense",
        action="store_true",
        help="predict unpooled spatiotemporal tokens (L1)",
    )
    ap.add_argument(
        "--predictor",
        choices=["spacetime", "persite"],
        default="spacetime",
        help="L1 dense predictor: factorised space-time (v2) or legacy per-site",
    )
    ap.add_argument(
        "--vicreg",
        type=float,
        default=0.1,
        help="weight of VICReg anti-collapse regulariser on dense predictions",
    )
    ap.add_argument(
        "--no-unc-weight",
        action="store_true",
        help="disable uncertainty-weighted multi-task loss (use raw sum)",
    )
    ap.add_argument(
        "--no-residual",
        action="store_true",
        help="predict absolute future latents instead of delta-from-last",
    )
    ap.add_argument(
        "--l1-causal",
        action="store_true",
        default=True,
        help="use autoregressive causal (GPT-style) L1 predictor (default, best)",
    )
    ap.add_argument(
        "--no-l1-causal",
        action="store_false",
        dest="l1_causal",
        help="use query-token (parallel) L1 predictor instead",
    )
    ap.add_argument(
        "--query-mask",
        choices=["block_causal", "parallel"],
        default="parallel",
        help="parallel reproduces the legacy encoder-bypass query baseline; "
        "use block_causal for a contextual query predictor",
    )
    ap.add_argument("--spatial-keep", type=int, default=256)
    ap.add_argument("--split-aware", action="store_true", default=True)
    ap.add_argument("--no-split-aware", action="store_false", dest="split_aware")
    ap.add_argument("--domains", default="", help="comma domains e.g. laparo,gi")
    ap.add_argument(
        "--train-domains",
        default="",
        help="restrict training to these domains (zero-shot transfer eval on others)",
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--history", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--n-actions", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument(
        "--clip-len",
        type=int,
        default=0,
        help="override encoder clip_len (0 = use encoder default; lower T saves cache memory)",
    )
    ap.add_argument("--max-clips", type=int, default=400)
    ap.add_argument(
        "--train-subset",
        type=int,
        default=0,
        help="subsample cached train latents to N clips (for scale curves)",
    )
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--latents", default="", help="reuse a latents_cache.pt for fair ablations"
    )
    ap.add_argument(
        "--save-dense-cache",
        action="store_true",
        help="also save dense (N-token) latents; large. Default saves pooled.",
    )
    ap.add_argument(
        "--cache-only",
        action="store_true",
        help="encode + save the (dense) latent cache, then exit before training",
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-clips", type=int, default=8)
    return ap


if __name__ == "__main__":
    train(build_argparser().parse_args())
