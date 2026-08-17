"""Train the latent world model on top of a frozen V-JEPA encoder.

Encode clips into per-timestep latent sequences z_{1..t} with the (frozen) target
encoder, then train a dynamics predictor to forecast future latents from a short
history. Rolling it forward = "imagining" the endoscopic world in latent space.

    python -m endoworld.simulation.train_world_model \
        --vjepa outputs/vjepa/vjepa_epoch8.pt --epochs 20
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from endoworld.data.video_clips import EndoClipDataset
from endoworld.understanding.vjepa import VJEPA, VJEPAConfig
from endoworld.simulation.latent_world_model import build_predictor, WorldModelConfig


def load_vjepa(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = VJEPAConfig(**ck["cfg"])
    model = VJEPA(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def collate(batch):
    return torch.stack([b if torch.is_tensor(b) else torch.as_tensor(b) for b in batch])


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    if args.encoder == "vjepa2":
        from endoworld.understanding.vjepa2_hf import VJEPA2Encoder
        vjepa = VJEPA2Encoder(args.vjepa2_id, device=device)
        clip_len, image_size, tubelet, embed_dim = 64, vjepa.image_size, vjepa.tubelet, vjepa.embed_dim
    else:
        vjepa, vcfg = load_vjepa(args.vjepa, device)
        clip_len, image_size, tubelet, embed_dim = vcfg.clip_len, vcfg.image_size, vcfg.tubelet_size, vcfg.embed_dim

    t_steps = clip_len // tubelet
    history = min(args.history, t_steps - 1)
    horizon = t_steps - history
    print(f"[world] encoder={args.encoder} latent seq len={t_steps}, history={history}, "
          f"horizon={horizon}, dim={embed_dim}")

    ds = EndoClipDataset(args.manifest, clip_len=clip_len, stride=args.stride,
                         image_size=image_size, exclude=["EndoVis2019_ROBUST-MIS"])
    if args.max_clips and len(ds.clips) > args.max_clips:
        import random
        random.seed(0)
        ds.clips = random.sample(ds.clips, args.max_clips)
    print(f"[data] {len(ds)} clips")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, collate_fn=collate, drop_last=True)

    # ---- encode ALL clips ONCE (cache latents), then train the small predictor fast ----
    print("[cache] encoding clips to latents (one pass through the frozen encoder) ...")
    Z = []
    import time as _t
    t0 = _t.time()
    for bi, clip in enumerate(dl):
        clip = clip.to(device).float()
        with torch.no_grad():
            Z.append(vjepa.encode_temporal(clip).cpu())
        if (bi + 1) % 10 == 0:
            print(f"  encoded {sum(z.shape[0] for z in Z)} clips ({_t.time()-t0:.0f}s)")
    Z = torch.cat(Z)                                     # (N, t, D)
    print(f"[cache] latents {tuple(Z.shape)} in {_t.time()-t0:.0f}s")

    n = Z.size(0)
    n_val = max(1, int(0.15 * n))
    perm = torch.randperm(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Z_tr, Z_val = Z[tr_idx].to(device), Z[val_idx].to(device)

    wcfg = WorldModelConfig(latent_dim=embed_dim, hidden_dim=args.hidden,
                            history=history, horizon=horizon)
    predictor = build_predictor(wcfg).to(device)
    print(f"[model] predictor {sum(p.numel() for p in predictor.parameters())/1e6:.2f}M params")
    opt = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=0.01)
    os.makedirs(args.out, exist_ok=True)
    bs = args.batch_size

    for epoch in range(args.epochs):
        predictor.train()
        idx = torch.randperm(Z_tr.size(0), device=device)
        run = 0.0
        for i in range(0, Z_tr.size(0), bs):
            z = Z_tr[idx[i:i + bs]]
            z_hist, z_future = z[:, :history], z[:, history:]
            pred = predictor(z_hist)
            loss = F.smooth_l1_loss(pred, z_future)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item() * z.size(0)
        if (epoch + 1) % max(args.epochs // 10, 1) == 0 or epoch == 0:
            print(f"[epoch {epoch}] train_loss={run/Z_tr.size(0):.4f}")

    # held-out rollout vs persistence baseline
    predictor.eval()
    with torch.no_grad():
        z_hist, z_future = Z_val[:, :history], Z_val[:, history:]
        pred = predictor(z_hist)
        persist = z_hist[:, -1:].expand_as(z_future)
        cos_model = F.cosine_similarity(pred, z_future, dim=-1).mean().item()
        cos_base = F.cosine_similarity(persist, z_future, dim=-1).mean().item()
        l2_model = (pred - z_future).pow(2).mean().item()
        l2_base = (persist - z_future).pow(2).mean().item()
    print(f"[rollout|val] cos_sim  model={cos_model:.3f}  persistence={cos_base:.3f}")
    print(f"[rollout|val] mse      model={l2_model:.4f}  persistence={l2_base:.4f}")

    torch.save({"predictor": predictor.state_dict(), "wcfg": wcfg.__dict__},
               os.path.join(args.out, "world_model.pt"))
    print(f"[ckpt] {os.path.join(args.out, 'world_model.pt')}")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["scratch", "vjepa2"], default="scratch")
    ap.add_argument("--vjepa", default="outputs/vjepa/vjepa_epoch8.pt")
    ap.add_argument("--vjepa2-id", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--manifest", default="manifests/sequences.csv")
    ap.add_argument("--out", default="outputs/world_model")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--history", type=int, default=4)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-clips", type=int, default=400)
    ap.add_argument("--workers", type=int, default=2)
    return ap


if __name__ == "__main__":
    train(build_argparser().parse_args())
