"""Crop domain-matched input/output panels for Figure 1.

figure10a rows (alphabetical): 0 CholecT50, 1 EndoNeRF, 2 HyperKvasir,
3 ION_bronch, 4 Kvasir-Capsule, 5 SCARED. Columns: last observed | GT |
retrieval | error. We crop (last observed, retrieval) per domain so each
Figure 1 row shows a true input->output pair from the same dataset.
"""
from pathlib import Path

from PIL import Image

FIG = Path(__file__).resolve().parent / "figures"

im = Image.open(FIG / "figure10a_forecast_qualitative.png")
w, h = im.size
top, row_h = 230, (h - 180) / 6
col_w = w / 4
pad_x = int(col_w * 0.10)  # drop row-label text and panel margins
rows = {"laparo": 0, "gi": 2, "bronch": 3}
for name, r in rows.items():
    y0 = int(top + r * row_h + 20)
    y1 = int(top + (r + 1) * row_h - 60)
    inp = im.crop((pad_x, y0, int(col_w) - pad_x, y1))
    inp.save(FIG / f"_fig1_in_{name}.png")
    out = im.crop((int(2 * col_w) + pad_x, y0, int(3 * col_w) - pad_x, y1))
    out.save(FIG / f"_fig1_out_{name}.png")

# SCARED physical-lane input: observed frame from figure8 row 0
im8 = Image.open(FIG / "figure8_qualitative.png")
w8, h8 = im8.size
top8, row_h8 = 120, (h8 - 160) / 3
col_w8 = w8 / 5
scared_in = im8.crop((0, int(top8), int(col_w8), int(top8 + row_h8)))
scared_in.save(FIG / "_fig1_in_scared.png")
print("[thumbs] wrote domain-matched pairs + SCARED input")
