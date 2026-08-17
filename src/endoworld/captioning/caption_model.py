"""Image -> text captioner (LLaVA-style: vision encoder -> projector -> decoder LM).

Self-contained and runnable offline: a compact from-scratch CNN vision encoder
produces spatial tokens, a projector maps them into the decoder's embedding space,
and a Transformer decoder generates the caption while cross-attending to vision
tokens. Swap `CNNVisionEncoder` for the pretrained V-JEPA encoder in production.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNVisionEncoder(nn.Module):
    """Small conv stack -> (B, num_tokens, dim) spatial tokens."""

    def __init__(self, dim: int = 256):
        super().__init__()
        def block(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1),
                                 nn.BatchNorm2d(o), nn.GELU())
        self.stem = nn.Sequential(
            block(3, 32, 2), block(32, 64, 2), block(64, 128, 2),
            block(128, dim, 2), block(dim, dim, 2),
        )  # 224 -> 7x7 tokens

    def forward(self, x):                      # x: (B, 3, H, W)
        f = self.stem(x)                       # (B, dim, h, w)
        return f.flatten(2).transpose(1, 2)    # (B, h*w, dim)


class DecoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n3 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x, mem, attn_mask):
        y = self.n1(x)
        x = x + self.self_attn(y, y, y, attn_mask=attn_mask, need_weights=False)[0]
        y = self.n2(x)
        x = x + self.cross_attn(y, mem, mem, need_weights=False)[0]
        x = x + self.mlp(self.n3(x))
        return x


class Captioner(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 256, heads: int = 4,
                 depth: int = 4, max_len: int = 64, vision: nn.Module | None = None):
        super().__init__()
        self.vision = vision or CNNVisionEncoder(dim)
        self.projector = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.tok_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([DecoderBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.max_len = max_len

    def encode_image(self, images):
        return self.projector(self.vision(images))

    def forward(self, images, tokens):
        """Teacher-forced training. tokens: (B, L) input ids (incl. BOS)."""
        mem = self.encode_image(images)
        b, L = tokens.shape
        x = self.tok_embed(tokens) + self.pos_embed[:, :L]
        causal = torch.triu(torch.ones(L, L, device=tokens.device) * float("-inf"), diagonal=1)
        for blk in self.blocks:
            x = blk(x, mem, causal)
        return self.head(self.norm(x))         # (B, L, vocab)

    @torch.no_grad()
    def generate(self, images, bos_id, eos_id, max_len: int | None = None):
        max_len = max_len or self.max_len
        mem = self.encode_image(images)
        b = images.size(0)
        seq = torch.full((b, 1), bos_id, dtype=torch.long, device=images.device)
        for _ in range(max_len - 1):
            L = seq.size(1)
            x = self.tok_embed(seq) + self.pos_embed[:, :L]
            causal = torch.triu(torch.ones(L, L, device=images.device) * float("-inf"), diagonal=1)
            for blk in self.blocks:
                x = blk(x, mem, causal)
            logits = self.head(self.norm(x))[:, -1]
            nxt = logits.argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
            if (nxt == eos_id).all():
                break
        return seq
