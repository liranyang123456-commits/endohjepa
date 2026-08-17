"""Download public GI datasets (HyperKvasir, Kvasir-Capsule, C3VD).

Resumable HTTP downloads. C3VD files are hosted on Google Drive; we parse the
project page for file ids and use the `confirm=t` export URL.
"""
from __future__ import annotations

import argparse
import re
import ssl
import time
import urllib.request
import zipfile
from pathlib import Path

from endoworld.data.prepare import extract_all_videos

REPO = Path(__file__).resolve().parents[3]
DATASETS = REPO / "datasets"
DL_ROOT = DATASETS / "_downloads"

UA = "Mozilla/5.0 (compatible; EndoWorld/1.0; research dataset fetch)"
CTX = ssl.create_default_context()

HYPERKVASIR = {
    "hyper-kvasir-videos.zip": "https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-videos.zip",
    "hyper-kvasir-labeled-images.zip": "https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip",
}
CAPSULE = {
    "kvasir-capsule-labeled-videos.zip": "https://datasets.simula.no/downloads/kvasir-capsule/kvasir-capsule-labeled-videos.zip",
    "kvasir-capsule-labeled-images.zip": "https://datasets.simula.no/downloads/kvasir-capsule/kvasir-capsule-labeled-images.zip",
    "kvasir-capsule-unlabeled-videos.zip": "https://datasets.simula.no/downloads/kvasir-capsule/kvasir-capsule-unlabeled-videos.zip",
}
C3VD_PAGE = "https://durrlab.github.io/C3VD/"


def _opener() -> urllib.request.OpenerDirector:
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", UA)]
    return opener


def download_http(url: str, dest: Path, chunk: int = 8 * 1024 * 1024) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": UA}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(f"[resume] {dest.name} from {existing / 1e9:.2f} GB")
    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if existing else "wb"
    t0 = time.time()
    last_log = t0
    with urllib.request.urlopen(req, timeout=120, context=CTX) as resp, open(tmp, mode) as f:
        total = existing
        clen = resp.headers.get("Content-Length")
        expected = existing + int(clen) if clen else None
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(buf)
            total += len(buf)
            now = time.time()
            if now - last_log > 5:
                gb = total / 1e9
                extra = f" / {expected / 1e9:.2f} GB" if expected else ""
                speed = (total - existing) / max(now - t0, 1) / 1e6
                print(f"  {dest.name}: {gb:.2f}{extra}  {speed:.1f} MB/s")
                last_log = now
    tmp.replace(dest)
    print(f"[ok] {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
    return dest


def unzip_to(zp: Path, out_dir: Path) -> None:
    flag = out_dir / ".extracted_ok"
    if flag.exists():
        print(f"[skip unzip] {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] {zp.name} -> {out_dir}")
    try:
        with zipfile.ZipFile(zp) as z:
            z.extractall(out_dir)
    except zipfile.BadZipFile:
        print(f"[bad zip] {zp} (size={zp.stat().st_size}); delete and re-download")
        zp.unlink(missing_ok=True)
        raise
    flag.write_text("ok", encoding="utf-8")


def fetch_hyperkvasir(include_images: bool = True) -> None:
    dest_ds = DATASETS / "HyperKvasir"
    dest_ds.mkdir(parents=True, exist_ok=True)
    items = dict(HYPERKVASIR)
    if not include_images:
        items.pop("hyper-kvasir-labeled-images.zip", None)
    for name, url in items.items():
        zp = DL_ROOT / name
        if not zp.exists():
            download_http(url, zp)
        sub = dest_ds / name.replace(".zip", "")
        unzip_to(zp, sub)
    print("[hyperkvasir] extracting video frames at 2 fps ...")
    extract_all_videos(str(dest_ds), target_fps=2.0, crop=True)


def fetch_capsule(include_unlabeled: bool = True) -> None:
    dest_ds = DATASETS / "Kvasir-Capsule"
    dest_ds.mkdir(parents=True, exist_ok=True)
    items = dict(CAPSULE)
    if not include_unlabeled:
        items.pop("kvasir-capsule-unlabeled-videos.zip", None)
    for name, url in items.items():
        zp = DL_ROOT / name
        if not zp.exists():
            download_http(url, zp)
        sub = dest_ds / name.replace(".zip", "")
        unzip_to(zp, sub)
    print("[capsule] extracting video frames at 2 fps ...")
    extract_all_videos(str(dest_ds), target_fps=2.0, crop=True)


def _c3vd_drive_files() -> list[tuple[str, str]]:
    req = urllib.request.Request(C3VD_PAGE, headers={"User-Agent": UA})
    html = urllib.request.urlopen(req, timeout=40, context=CTX).read().decode("utf-8", "replace")
    pairs = re.findall(
        r"id=([A-Za-z0-9_-]+)&amp;confirm=t\"><span[^>]*>([A-Za-z0-9_.]+\.zip)",
        html,
    )
    keep_prefix = ("cecum_", "desc_", "sigmoid_", "trans_", "screening_")
    seen: dict[str, str] = {}
    for fid, name in pairs:
        if name.startswith(keep_prefix):
            seen[name] = fid
    return [(n, i) for n, i in seen.items()]


def download_gdrive(file_id: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError:
        download_http(
            f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t", dest)
        return dest
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[gdown] {dest.name}")
    gdown.download(url, str(dest), quiet=False, fuzzy=True)
    return dest


def fetch_c3vd(max_files: int | None = None, skip_screening: bool = False) -> None:
    dest_ds = DATASETS / "C3VD"
    dest_ds.mkdir(parents=True, exist_ok=True)
    files = _c3vd_drive_files()
    if skip_screening:
        files = [(n, i) for n, i in files if not n.startswith("screening")]
    if max_files:
        files = files[:max_files]
    print(f"[c3vd] {len(files)} google-drive zips")
    for name, fid in files:
        zp = DL_ROOT / "c3vd" / name
        if not zp.exists():
            try:
                download_gdrive(fid, zp)
            except Exception as e:
                print(f"[c3vd] failed {name}: {e}")
                continue
        if zp.exists() and zp.stat().st_size > 10_000_000:
            unzip_to(zp, dest_ds / name.replace(".zip", ""))
        else:
            print(f"[c3vd] skip unzip, file too small (likely gdrive html): {zp}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["hyperkvasir", "capsule", "c3vd", "all"], default="all")
    ap.add_argument("--no-unlabeled-capsule", action="store_true")
    ap.add_argument("--c3vd-max", type=int, default=None)
    ap.add_argument("--skip-screening", action="store_true")
    args = ap.parse_args()
    DL_ROOT.mkdir(parents=True, exist_ok=True)
    if args.set in ("hyperkvasir", "all"):
        fetch_hyperkvasir()
    if args.set in ("capsule", "all"):
        fetch_capsule(include_unlabeled=not args.no_unlabeled_capsule)
    if args.set in ("c3vd", "all"):
        fetch_c3vd(max_files=args.c3vd_max, skip_screening=args.skip_screening)


if __name__ == "__main__":
    main()
