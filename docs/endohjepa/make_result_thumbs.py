"""Crop representative result panels from the qualitative montages for Figure 1."""
from pathlib import Path

from PIL import Image

FIG = Path(__file__).resolve().parent / "figures"

# figure10a: 6 rows x 4 cols; row 0 = CholecT50 (last observed | GT | retrieval | error)
im = Image.open(FIG / "figure10a_forecast_qualitative.png")
w, h = im.size
top, row_h = 230, (h - 180) / 6
col_w = w / 4
gt = im.crop((int(col_w), int(top), int(2 * col_w), int(top + row_h - 40)))
gt.save(FIG / "_fig1_forecast_gt.png")
ret = im.crop((int(2 * col_w), int(top), int(3 * col_w), int(top + row_h - 40)))
ret.save(FIG / "_fig1_forecast_ret.png")

# figure8: 3 rows x 5 cols; row 0 (observed | GT | SE3 retrieval | shuffled | error)
im8 = Image.open(FIG / "figure8_qualitative.png")
w8, h8 = im8.size
top8, row_h8 = 120, (h8 - 160) / 3
col_w8 = w8 / 5
se3 = im8.crop((int(2 * col_w8), int(top8), int(3 * col_w8), int(top8 + row_h8)))
se3.save(FIG / "_fig1_se3.png")

# figure11: 4 rows x 5 cols; row 1 = model rollout
im11 = Image.open(FIG / "figure11_rollout_strip.png")
w11, h11 = im11.size
top11, row_h11 = 110, (h11 - 150) / 4
col_w11 = w11 / 5
nav = im11.crop((int(3 * col_w11), int(top11 + row_h11), int(4 * col_w11), int(top11 + 2 * row_h11)))
nav.save(FIG / "_fig1_nav.png")
print("[thumbs] wrote 4 result thumbnails")
