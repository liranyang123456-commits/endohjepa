"""Build one clean, self-contained MedIA submission package."""

from __future__ import annotations

import hashlib
import json
import re
import runpy
import shutil
import zipfile
from datetime import date
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE / "submission_package"
PACKAGE_NAME = "MedIA_EndoHJEPA_submission_20260820"
OUT = PACKAGE_ROOT / PACKAGE_NAME
ZIP_PATH = PACKAGE_ROOT / f"{PACKAGE_NAME}.zip"


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    runpy.run_path(str(HERE / "sanitize_provenance.py"), run_name="__main__")
    if PACKAGE_ROOT.exists():
        shutil.rmtree(PACKAGE_ROOT)
    OUT.mkdir(parents=True)

    core_files = [
        "endohjepa.tex",
        "endohjepa.pdf",
        "references.bib",
        "highlights.txt",
        "cover_letter.md",
        "declarations.txt",
        "ION_ETHICS_TEMPLATE.md",
        "graphical_abstract.txt",
        "verified_metrics.json",
        "c3vd_pose_gate.json",
        "audit_contact_protocol.json",
    ]
    for name in core_files:
        _copy(HERE / name, OUT / name)

    manuscript = (HERE / "endohjepa.tex").read_text(encoding="utf-8")
    figure_names = {
        Path(name).name
        for name in re.findall(
            r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", manuscript
        )
    }
    for name in sorted(figure_names):
        source = HERE / "figures" / name
        _copy(source, OUT / "figures" / name)

    for name in ("graphical_abstract.pdf", "graphical_abstract.png"):
        _copy(HERE / "figures" / name, OUT / name)

    provenance_files = [
        "dataset_atlas_provenance.json",
        "figure1_provenance.json",
        "figure8_qualitative_provenance.json",
        "figure11_rollout_provenance.json",
        "graphical_abstract_provenance.json",
        "qualitative_forecast_15000.json",
    ]
    for name in provenance_files:
        _copy(HERE / name, OUT / "provenance" / name)

    readme = f"""Medical Image Analysis submission package
Generated: {date.today().isoformat()}

Upload:
1. endohjepa.pdf as the manuscript.
2. endohjepa.tex, references.bib and figures/ as LaTeX source files.
3. highlights.txt.
4. cover_letter.md.
5. declarations.txt.
6. graphical_abstract.pdf or graphical_abstract.png.

The provenance/ directory and verified_metrics.json provide the auditable
result records used by the manuscript. Private ION data are not included.
"""
    (OUT / "README_UPLOAD.txt").write_text(readme, encoding="utf-8")

    blockers = []
    if (
        "must be supplied by the responsible institution before submission"
        in manuscript
    ):
        blockers.append(
            "Confirm the private-cohort review committee name, ethics approval "
            "or exemption identifier, decision date and consent-waiver determination; "
            "then replace the explicit unresolved statement in endohjepa.tex, "
            "declarations.txt and the cover letter as applicable."
        )
    if blockers:
        (OUT / "SUBMISSION_BLOCKERS.txt").write_text(
            "Do not submit until all items below are resolved:\n\n"
            + "\n".join(f"{index}. {item}" for index, item in enumerate(blockers, 1))
            + "\n",
            encoding="utf-8",
        )

    manifest = {
        "package": PACKAGE_NAME,
        "generated": date.today().isoformat(),
        "files": {},
    }
    for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
        relative = path.relative_to(OUT).as_posix()
        manifest["files"][relative] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    (OUT / "SUBMISSION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(OUT).as_posix())

    print(f"[submission] {OUT}")
    print(f"[submission] {ZIP_PATH}")
    print(f"[submission] {len(manifest['files']) + 1} files")


if __name__ == "__main__":
    main()
