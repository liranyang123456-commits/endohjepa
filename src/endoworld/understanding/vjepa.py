"""V-JEPA style video self-supervised model (compact, trainable).

Architecture (LeCun's JEPA in latent space):
  - Tubelet embedding: Conv3d turns a clip (B,C,T,H,W) into spatio-temporal tokens.
  - Context encoder (ViT): encodes only the *visible* tokens.
  - Target encoder (EMA copy of a full ViT): encodes *all* tokens; provides targets.
  - Predictor (smaller ViT): from visible tokens + mask tokens at masked positions,
    predicts the target representations of the masked tokens.
  - Loss: SmoothL1 between predicted and target (stop-grad) latents at masked positions.

This is a from-scratch reference that trains end-to-end. For SOTA, load official
V-JEPA 2 weights into the same interface and domain-adapt.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class VJEPAConfig:
    image_size: int = 224
    clip_len: int = 16
    in_chans: int = 3
    patch_size: int = 16
    tubelet_size: int = 2
    embed_dim: int = 384  # compact ViT (vit_small-ish)
    depth: int = 6
    num_heads: int = 6
    predictor_dim: int = 192
    predictor_depth: int = 4
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.75
    ema_momentum: float = 0.996

    @property
    def n_tokens(self) -> int:
        t = self.clip_len // self.tubelet_size
        hw = (self.image_size // self.patch_size) ** 2
        return t * hw


def _sincos_pos_embed(n_tokens: int, dim: int) -> torch.Tensor:
    pos = torch.arange(n_tokens).float().unsqueeze(1)
    i = torch.arange(dim // 2).float().unsqueeze(0)
    freq = torch.exp(-math.log(10000.0) * (2 * i) / dim)
    ang = pos * freq
    pe = torch.zeros(n_tokens, dim)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)
    return pe.unsqueeze(0)


class TubeletEmbed(nn.Module):
    def __init__(self, cfg: VJEPAConfig):
        super().__init__()
        self.proj = nn.Conv3d(
            cfg.in_chans,
            cfg.embed_dim,
            kernel_size=(cfg.tubelet_size, cfg.patch_size, cfg.patch_size),
            stride=(cfg.tubelet_size, cfg.patch_size, cfg.patch_size),
        )

    def forward(self, x):  # x: (B, C, T, H, W)
        x = self.proj(x)  # (B, D, t, h, w)
        return x.flatten(2).transpose(1, 2)  # (B, N, D)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x):
        y = self.n1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        x = x + self.mlp(self.n2(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, cfg: VJEPAConfig):
        super().__init__()
        self.patch = TubeletEmbed(cfg)
        self.register_buffer("pos", _sincos_pos_embed(cfg.n_tokens, cfg.embed_dim))
        self.blocks = nn.ModuleList(
            [
                Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio)
                for _ in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward_tokens(self, tokens):
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)

    def forward(self, x, keep_idx=None):
        tok = self.patch(x) + self.pos  # (B, N, D)
        if keep_idx is not None:
            b, n, d = tok.shape
            idx = keep_idx.unsqueeze(-1).expand(-1, -1, d)
            tok = torch.gather(tok, 1, idx)  # keep only visible
        return self.forward_tokens(tok)


class Predictor(nn.Module):
    def __init__(self, cfg: VJEPAConfig):
        super().__init__()
        self.embed = nn.Linear(cfg.embed_dim, cfg.predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.predictor_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer("pos", _sincos_pos_embed(cfg.n_tokens, cfg.predictor_dim))
        self.blocks = nn.ModuleList(
            [
                Block(cfg.predictor_dim, cfg.num_heads, cfg.mlp_ratio)
                for _ in range(cfg.predictor_depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.predictor_dim)
        self.head = nn.Linear(cfg.predictor_dim, cfg.embed_dim)

    def forward(self, vis_tokens, vis_idx, mask_idx):
        b = vis_tokens.size(0)
        self.pos.size(1)
        x = self.embed(vis_tokens) + self._gather_pos(vis_idx)
        mask = self.mask_token.expand(b, mask_idx.size(1), -1) + self._gather_pos(
            mask_idx
        )
        seq = torch.cat([x, mask], dim=1)
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm(seq)
        pred_mask = seq[:, x.size(1) :, :]  # predictions at masked positions
        return self.head(pred_mask)

    def _gather_pos(self, idx):
        d = self.pos.size(-1)
        return torch.gather(
            self.pos.expand(idx.size(0), -1, -1), 1, idx.unsqueeze(-1).expand(-1, -1, d)
        )


class VJEPA(nn.Module):
    def __init__(self, cfg: VJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.context_encoder = ViTEncoder(cfg)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(cfg)

    @torch.no_grad()
    def update_target(self):
        m = self.cfg.ema_momentum
        for tp, cp in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            tp.data.mul_(m).add_(cp.data, alpha=1 - m)

    def sample_masks(self, batch: int, device):
        n = self.cfg.n_tokens
        n_mask = int(n * self.cfg.mask_ratio)
        vis_list, mask_list = [], []
        for _ in range(batch):
            perm = torch.randperm(n, device=device)
            mask_list.append(perm[:n_mask])
            vis_list.append(perm[n_mask:])
        return torch.stack(vis_list), torch.stack(mask_list)

    def forward(self, clip, token_weights=None):
        # clip: (B, T, C, H, W) -> (B, C, T, H, W)
        # token_weights: optional (B, N) keep-weights (specular down-weight)
        x = clip.permute(0, 2, 1, 3, 4).contiguous()
        b = x.size(0)
        vis_idx, mask_idx = self.sample_masks(b, x.device)

        vis_tokens = self.context_encoder(x, keep_idx=vis_idx)
        pred = self.predictor(vis_tokens, vis_idx, mask_idx)

        with torch.no_grad():
            full = self.target_encoder(x)  # (B, N, D)
            d = full.size(-1)
            tgt = torch.gather(full, 1, mask_idx.unsqueeze(-1).expand(-1, -1, d))
        per = nn.functional.smooth_l1_loss(pred, tgt, reduction="none").mean(dim=-1)
        if token_weights is None:
            return per.mean()
        w = torch.gather(token_weights, 1, mask_idx).clamp_min(0)
        return (per * w).sum() / w.sum().clamp_min(1e-6)

    @torch.no_grad()
    def encode(self, clip):
        """Return pooled clip representation (for probing / downstream)."""
        x = clip.permute(0, 2, 1, 3, 4).contiguous()
        tokens = self.target_encoder(x)
        return tokens.mean(dim=1)

    @torch.no_grad()
    def encode_dense(self, clip):
        """Spatio-temporal tokens (B, t, hw, D) without spatial pooling."""
        x = clip.permute(0, 2, 1, 3, 4).contiguous()
        tokens = self.target_encoder(x)
        t = self.cfg.clip_len // self.cfg.tubelet_size
        b, n, d = tokens.shape
        hw = n // t
        return tokens.view(b, t, hw, d)

    @torch.no_grad()
    def encode_temporal(self, clip):
        """Return a per-timestep latent sequence (B, t, D), pooled over space.

        Token order from the Conv3d tubelet embedding is t-major (t, h, w), so we
        reshape to (B, t, h*w, D) and average over the spatial axis.
        """
        return self.encode_dense(clip).mean(dim=2)
