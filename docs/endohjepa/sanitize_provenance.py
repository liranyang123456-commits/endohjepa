"""Remove workstation paths and private folder metadata from public records."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
FILES = [
    "action_retrieval_provenance.json",
    "dataset_atlas_provenance.json",
    "figure8_qualitative_provenance.json",
    "figure11_rollout_provenance.json",
    "figure1_provenance.json",
    "figure1_thumb_selection.json",
    "geometry_asset_gate.json",
    "recorded_rollout_provenance.json",
]
DRIVE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")


def _is_private_or_absolute_path(value: str) -> bool:
    normal = value.replace("\\", "/")
    return (
        DRIVE_PATH.search(value) is not None
        or normal.startswith("/home/")
        or normal.startswith("/Users/")
        or "datasets/ion_bronch/" in normal.lower()
    )


def _public_path(value: str, dataset: str | None = None) -> str:
    normal = value.replace("\\", "/")
    lower = normal.lower()
    if "datasets/ion_bronch/" in lower:
        tail = normal[lower.index("datasets/ion_bronch/") :]
        case = next(
            (part for part in tail.split("/") if re.fullmatch(r"case_\d+", part)),
            "case_XXX",
        )
        return f"datasets/ION_bronch/{case}/anonymised_sequence/{Path(normal).name}"
    if "scared/" in lower:
        return "datasets/SCARED/" + normal[lower.index("scared/") + len("scared/") :]
    dataset_root = re.search(r"(?i)(?:^|/)datasets/(.+)", normal)
    if dataset_root:
        return f"datasets/{dataset_root.group(1)}"
    if "manifests/" in lower:
        return normal[lower.index("manifests/") :]
    if dataset:
        return f"external/{dataset}/{Path(normal).name}"
    return f"external/{Path(normal).name}"


def _sanitize(node: Any, dataset: str | None = None) -> Any:
    if isinstance(node, dict):
        local_dataset = str(node.get("dataset", dataset or "")) or dataset
        cleaned = {}
        for key, value in node.items():
            key = key.replace("Datasets/SCARED", "datasets/SCARED")
            public_key = (
                _public_path(key, local_dataset) if DRIVE_PATH.search(key) else key
            )
            if key == "manifest_sequence_id" and local_dataset == "ION_bronch":
                case = re.search(r"case_\d+", str(value))
                cleaned[public_key] = (
                    f"{case.group(0)}/anonymised_sequence"
                    if case
                    else "case_XXX/anonymised_sequence"
                )
            elif isinstance(value, str) and _is_private_or_absolute_path(value):
                cleaned[public_key] = _public_path(value, local_dataset)
            else:
                cleaned[public_key] = _sanitize(value, local_dataset)
        return cleaned
    if isinstance(node, list):
        return [_sanitize(value, dataset) for value in node]
    if isinstance(node, str):
        node = node.replace("Datasets/SCARED", "datasets/SCARED")
        if _is_private_or_absolute_path(node):
            return _public_path(node, dataset)
    return node


def main() -> None:
    for name in FILES:
        path = HERE / name
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if name == "figure1_provenance.json":
            data["figure"] = [
                "figures/figure1_pipeline.pdf",
                "figures/figure1_pipeline.png",
            ]
        path.write_text(json.dumps(_sanitize(data), indent=2), encoding="utf-8")
        print(f"[sanitize] {path.name}")


if __name__ == "__main__":
    main()
