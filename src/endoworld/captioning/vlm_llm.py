"""LLaVA-style VLM: vision encoder -> projector -> real LLM (Qwen2.5).

The LLM is kept frozen (LLaVA stage-1 alignment); we train a vision encoder + a
projector that maps visual tokens into the LLM embedding space so the frozen LLM can
generate fluent captions conditioned on the image.

    python -m endoworld.captioning.vlm_llm --epochs 3 --limit 2000
    python -m endoworld.captioning.vlm_llm --smoke
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from endoworld.captioning.caption_model import CNNVisionEncoder

LLM_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Describe this endoscopic image:"


class VLM(nn.Module):
    def __init__(self, llm_id=LLM_ID, device="cuda"):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(llm_id)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_id, dtype=torch.float32).to(
            device
        )
        for p in self.llm.parameters():
            p.requires_grad_(False)
        self.llm.eval()
        d_llm = self.llm.config.hidden_size
        self.vision = CNNVisionEncoder(dim=256)
        self.projector = nn.Sequential(
            nn.Linear(256, d_llm), nn.GELU(), nn.Linear(d_llm, d_llm)
        )
        self.device = device
        self.embed = self.llm.get_input_embeddings()

    def visual_tokens(self, images):
        return self.projector(self.vision(images))  # (B, K, d_llm)

    def _text_embeds(self, ids):
        return self.embed(ids)

    def forward(self, images, cap_ids, cap_mask):
        vis = self.visual_tokens(images)  # (B,K,D)
        b, K, D = vis.shape
        prompt_ids = self.tok(PROMPT, return_tensors="pt").input_ids.to(self.device)
        prompt_emb = self._text_embeds(prompt_ids).expand(b, -1, -1)
        cap_emb = self._text_embeds(cap_ids)  # (B,L,D)
        inp = torch.cat([vis, prompt_emb, cap_emb], dim=1)

        pre = K + prompt_emb.size(1)
        labels = torch.full(
            (b, inp.size(1)), -100, dtype=torch.long, device=self.device
        )
        labels[:, pre:] = torch.where(
            cap_mask.bool(), cap_ids, torch.full_like(cap_ids, -100)
        )
        attn = torch.ones(inp.size(0), inp.size(1), device=self.device)
        attn[:, pre:] = cap_mask.float()
        out = self.llm(inputs_embeds=inp, attention_mask=attn, labels=labels)
        return out.loss

    @torch.no_grad()
    def generate(self, images, max_new_tokens=40):
        vis = self.visual_tokens(images)
        b = vis.size(0)
        prompt_ids = self.tok(PROMPT, return_tensors="pt").input_ids.to(self.device)
        prompt_emb = self._text_embeds(prompt_ids).expand(b, -1, -1)
        inp = torch.cat([vis, prompt_emb], dim=1)
        attn = torch.ones(inp.size(0), inp.size(1), device=self.device)
        gen = self.llm.generate(
            inputs_embeds=inp,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tok.eos_token_id,
        )
        return [self.tok.decode(g, skip_special_tokens=True) for g in gen]


class PairDS(Dataset):
    def __init__(self, rows, tok, image_size=224, max_len=48):
        self.rows, self.tok, self.image_size, self.max_len = (
            rows,
            tok,
            image_size,
            max_len,
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        path, cap = self.rows[i]
        try:
            img = (
                Image.open(path)
                .convert("RGB")
                .resize((self.image_size, self.image_size))
            )
            arr = np.asarray(img, np.float32) / 255.0
        except Exception:
            arr = np.zeros((self.image_size, self.image_size, 3), np.float32)
        img_t = torch.from_numpy(arr).permute(2, 0, 1)
        enc = self.tok(
            cap + self.tok.eos_token,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return img_t, enc.input_ids[0], enc.attention_mask[0]


def load_pairs(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)
        return [(row[0], row[1]) for row in r if len(row) >= 2]


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    model = VLM(args.llm, device).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(
        f"[model] LLM={args.llm} (frozen); trainable={n_train:.1f}M (vision+projector)"
    )

    rows = load_pairs(args.pairs)
    if args.smoke:
        rows = rows[:32]
    if args.limit:
        rows = rows[: args.limit]
    img_size = 96 if args.smoke else 224
    ds = PairDS(rows, model.tok, image_size=img_size)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    print(f"[data] {len(ds)} pairs")

    params = list(model.vision.parameters()) + list(model.projector.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    os.makedirs(args.out, exist_ok=True)

    for epoch in range(args.epochs):
        model.vision.train()
        model.projector.train()
        run = 0.0
        for img, ids, mask in dl:
            img, ids, mask = img.to(device), ids.to(device), mask.to(device)
            loss = model(img, ids, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item()
        print(f"[epoch {epoch}] loss={run / max(len(dl), 1):.4f}")

    model.eval()
    img, ids, _ = ds[0]
    pred = model.generate(img.unsqueeze(0).to(device))[0]
    print("[sample] GT  :", model.tok.decode(ids, skip_special_tokens=True))
    print("[sample] PRED:", pred.strip())

    torch.save(
        {
            "vision": model.vision.state_dict(),
            "projector": model.projector.state_dict(),
            "llm": args.llm,
        },
        os.path.join(args.out, "vlm_llm.pt"),
    )
    print(f"[ckpt] {os.path.join(args.out, 'vlm_llm.pt')}")
    if args.smoke:
        print("[smoke] OK")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="manifests/caption_pairs_cholecseg8k.csv")
    ap.add_argument("--llm", default=LLM_ID)
    ap.add_argument("--out", default="outputs/vlm_llm")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    return ap


if __name__ == "__main__":
    train(build_argparser().parse_args())
