"""Extract structured nodule / anatomy parameters from the clinical record text.

The provincial-hospital ION records use very regular Chinese phrasing, so we regex
out the planning-relevant fields (nodule size, lobe, region, airway segment/generation,
distances to pleura / chest wall / vessel, nodule solidity, malignancy probability).
These are exactly the inputs an ablation planner needs (and serve as ground truth).

    python -m endoworld.ablation.parse_records \
        --notes clinical_staging/notes_text --out manifests/nodule_params.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re

NUM = r"([0-9]+(?:\.[0-9]+)?)"


def _find(pat, text, cast=float, default=None):
    m = re.search(pat, text)
    if not m:
        return default
    try:
        return cast(m.group(1))
    except (ValueError, IndexError):
        return m.group(1) if m.groups() else default


def parse_note(text: str) -> dict:
    t = text.replace(" ", "")  # records are full of stray spaces
    d = {}
    d["diam_axial_mm"] = _find(r"病变轴位直径" + NUM, t)
    d["diam_coronal_mm"] = _find(r"病变冠状面直径" + NUM, t)
    d["diam_sagittal_mm"] = _find(r"病变矢状面直径" + NUM, t)
    d["size_AP_mm"] = _find(r"AP[（(]?前后[）)]?[:：]" + NUM, t)
    d["size_SI_mm"] = _find(r"SI[（(]?上下[）)]?[:：]" + NUM, t)
    d["size_LR_mm"] = _find(r"LR[（(]?左右[）)]?[:：]" + NUM, t)
    # lobe
    m = re.search(r"病变部位位于(右上肺叶|右中肺叶|右下肺叶|左上肺叶|左下肺叶)", t)
    d["lobe"] = m.group(1) if m else None
    # region (outer/middle/inner third)
    m = re.search(r"位于(外|中|内)1/3肺区", t)
    d["region_third"] = {"外": "outer", "中": "middle", "内": "inner"}.get(m.group(1)) if m else None
    d["bronchial_segment"] = _find(r"支气管节段属于([A-Z]{1,2}B?[0-9]{1,2})", t, str)
    d["airway_generation"] = _find(r"气道分级数为第" + NUM + r"级", t, int)
    d["dist_pleura_mm"] = _find(r"与最近的胸膜或胸膜表面[（(]?即肺裂[）)]?的距离" + NUM, t) \
        or _find(r"胸膜.{0,10}?距离" + NUM, t)
    d["dist_chestwall_mm"] = _find(r"与最近的胸壁距离" + NUM, t)
    m = re.search(r"与最近重要血管的距离(&gt;|>|＞)?" + NUM, t)
    d["dist_vessel_mm"] = float(m.group(2)) if m else None
    d["dist_vessel_op"] = ">" if (m and m.group(1)) else "="
    m = re.search(r"结节类型为(实性|部分实性|磨玻璃|亚实性)", t)
    d["solidity"] = m.group(1) if m else None
    d["malignancy_pct"] = _find(r"恶性可能性百分比[:：]?" + NUM, t)
    d["n_planned_paths"] = _find(r"规划的导航路径数量[:：]" + NUM, t, int) \
        or _find(r"尝试的总导航路径数" + NUM, t, int)
    # demographics
    d["sex"] = _find(r"性别[:：](男|女)", t, str)
    d["height_cm"] = _find(r"身高[:：]" + NUM, t)
    d["weight_kg"] = _find(r"体重[:：]" + NUM, t)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="clinical_staging/notes_text")
    ap.add_argument("--out", default="manifests/nodule_params.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.notes, "*.txt")))
    rows = []
    for f in files:
        text = open(f, encoding="utf-8").read()
        rec = {"note": os.path.basename(f)}
        rec.update(parse_note(text))
        rows.append(rec)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = ["note", "sex", "height_cm", "weight_kg", "lobe", "region_third",
            "bronchial_segment", "airway_generation", "diam_axial_mm",
            "diam_coronal_mm", "diam_sagittal_mm", "size_AP_mm", "size_SI_mm",
            "size_LR_mm", "dist_pleura_mm", "dist_chestwall_mm", "dist_vessel_mm",
            "dist_vessel_op", "solidity", "malignancy_pct", "n_planned_paths"]
    with open(args.out, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})

    n_ok = sum(1 for r in rows if r.get("diam_axial_mm") or r.get("size_AP_mm"))
    print(f"parsed {len(rows)} notes, {n_ok} with nodule size -> {args.out}")
    return rows


if __name__ == "__main__":
    main()
