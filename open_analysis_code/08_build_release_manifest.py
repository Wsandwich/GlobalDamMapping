#!/usr/bin/env python3
"""Build a checksum manifest for public data files."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPOSITORY_ROOT / "metadata" / "release_manifest.csv"
DATA_GROUPS = {
    "aggregated_dam_statistics": "global_or_grouped_dam_summary",
    "anonymized_segment_dof": "schema_only_optional_segment_product",
    "ffr_aggregate_tables": "grouped_ffr_summary",
    "model_outputs": "model_summary",
    "figure_source_data": "nonspatial_figure_source_data",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> int | str:
    if path.suffix.lower() != ".csv":
        return ""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def main() -> int:
    rows: list[dict[str, object]] = []
    for directory, release_class in DATA_GROUPS.items():
        root = REPOSITORY_ROOT / directory
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == "README.md":
                continue
            relative = path.relative_to(REPOSITORY_ROOT)
            rows.append(
                {
                    "relative_path": relative.as_posix(),
                    "release_class": release_class,
                    "rows": csv_rows(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "exact_spatial_identifiers": "no",
                    "small_count_review": "required_and_audited",
                }
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} manifest entries to {OUTPUT.relative_to(REPOSITORY_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
