"""Sync submission package v3 with main directory and rebuild manifest hashes."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(".")
PKG = Path("submission_package/MedIA_EndoHJEPA_submission_final_v3")

n_fig = 0
for src in sorted((ROOT / "figures").glob("*")):
    if src.suffix.lower() in {".pdf", ".png"} and not src.name.startswith("_"):
        shutil.copy2(src, PKG / "figures" / src.name)
        n_fig += 1

for src in sorted(ROOT.glob("*")):
    if src.is_file() and src.suffix.lower() in {".tex", ".pdf", ".json", ".txt", ".md", ".bib"}:
        dst = PKG / src.name
        if dst.exists():
            shutil.copy2(src, dst)

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()

manifest_path = PKG / "SUBMISSION_MANIFEST.json"
old = json.loads(manifest_path.read_text(encoding="utf-8"))

files = []
for p in sorted(PKG.rglob("*")):
    if p.is_file() and p.name != "SUBMISSION_MANIFEST.json":
        files.append({"path": str(p.relative_to(PKG)).replace("\\", "/"),
                      "sha256": sha256(p)})

fig_count = len([f for f in files if f["path"].startswith("figures/")])
notes = old.get("notes", [])
closure = ("2026-08-22 audit: STIR entry switched to the patient-level grouped held-out "
           "split (177.7 -> 172.2, n=8 sequences / 2 held-out patients); adaptation-pool "
           "leakage caveat disclosed for the domain-adapted CholecT50 rows; test suite "
           "re-aligned with the refactored eval modules (26 passed); GitHub release synced.")
if closure not in notes:
    notes.append(closure)

manifest = {
    "package": old.get("package", "MedIA_EndoHJEPA_submission_final_v3"),
    "files": files,
    "figures_included": fig_count,
    "notes": notes,
}
manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                         encoding="utf-8")
print(f"synced {n_fig} figures; manifest lists {len(files)} files, {fig_count} figures")
