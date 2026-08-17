"""Train the image->text captioner on synthesized caption pairs.

    python -m endoworld.captioning.train_caption \
        --pairs manifests/caption_pairs_cholecseg8k.csv --epochs 10

    python -m endoworld.captioning.train_caption --smoke
"""
from __future__ import annotations

import argparse
import csv
import os
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from endoworld.captioning.caption_model import Captioner

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z\-]+|[.,:;()]", text.lower())


class Vocab:
    def __init__(self, texts, min_freq=1):
        from collections import Counter
        c = Counter(t for s in texts for t in tokenize(s))
        self.itos = list(SPECIAL) + [w for w, f in c.items() if f >= min_freq]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad, self.bos, self.eos, self.unk = 0, 1, 2, 3

    def __len__(self):
        return len(self.itos)

    def encode(self, text, max_len):
        ids = [self.bos] + [self.stoi.get(t, self.unk) for t in tokenize(text)] + [self.eos]
        ids = ids[:max_len]
        ids += [self.pad] * (max_len - len(ids))
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            if i == self.eos:
                break
            if i in (self.pad, self.bos):
                continue
            out.append(self.itos[i] if i < len(self.itos) else "<unk>")
        return " ".join(out)


class CaptionDataset(Dataset):
    def __init__(self, rows, vocab: Vocab, image_size=224, max_len=48):
        self.rows = rows
        self.vocab = vocab
        self.image_size = image_size
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        path, caption = self.rows[idx]
        try:
            img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0
        except Exception:
            arr = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        img_t = torch.from_numpy(arr).permute(2, 0, 1)
        ids = torch.tensor(self.vocab.encode(caption, self.max_len), dtype=torch.long)
        return img_t, ids


def load_pairs(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)
        return [(row[0], row[1]) for row in r if len(row) >= 2]


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    rows = load_pairs(args.pairs)
    if args.smoke:
        rows = rows[:64]
    print(f"[data] {len(rows)} pairs")

    vocab = Vocab([c for _, c in rows])
    print(f"[vocab] size={len(vocab)}")

    n_val = max(1, int(0.1 * len(rows)))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    img_size = 96 if args.smoke else args.image_size
    max_len = 48
    tr = CaptionDataset(train_rows, vocab, img_size, max_len)
    va = CaptionDataset(val_rows, vocab, img_size, max_len)
    dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, drop_last=True)

    dim = 128 if args.smoke else 256
    depth = 2 if args.smoke else 4
    model = Captioner(len(vocab), dim=dim, depth=depth, max_len=max_len).to(device)
    print(f"[model] {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    crit = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad)
    os.makedirs(args.out, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        run = 0.0
        for img, ids in dl:
            img, ids = img.to(device), ids.to(device)
            logits = model(img, ids[:, :-1])              # predict next token
            loss = crit(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item()
        print(f"[epoch {epoch}] loss={run/max(len(dl),1):.4f}")

    # qualitative eval
    model.eval()
    img, ids = va[0]
    gen = model.generate(img.unsqueeze(0).to(device), vocab.bos, vocab.eos)
    print("[sample] GT  :", vocab.decode(ids.tolist()))
    print("[sample] PRED:", vocab.decode(gen[0].tolist()))

    torch.save({"model": model.state_dict(), "vocab": vocab.itos,
                "dim": dim, "depth": depth, "max_len": max_len},
               os.path.join(args.out, "captioner.pt"))
    print(f"[ckpt] {os.path.join(args.out, 'captioner.pt')}")
    if args.smoke:
        print("[smoke] OK")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="manifests/caption_pairs_cholecseg8k.csv")
    ap.add_argument("--out", default="outputs/captioner")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    return ap


if __name__ == "__main__":
    train(build_argparser().parse_args())
