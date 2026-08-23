#!/usr/bin/env python3
"""Audit the repository for sensitive fields and GitHub release hazards."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from common.privacy import find_exact_identifiers, infer_count_columns, is_primary_small_count


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".cff", ".json"}
CODE_SUFFIXES = {".py"}
PROHIBITED_SUFFIXES = {
    ".shp",
    ".shx",
    ".dbf",
    ".prj",
    ".cpg",
    ".gpkg",
    ".tif",
    ".tiff",
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
    ".parquet",
}
STRICT_SEGMENT_FORBIDDEN = {
    "goid",
    "noid",
    "ndoid",
    "nuoid",
    "hybas_id",
    "geometry",
    "longitude",
    "latitude",
    "lon",
    "lat",
    "source_row",
    "upstream",
    "downstream",
    "distance_to_river",
    "dis_av_cms",
    "discharge",
    "exact_length",
    "dam_count",
    "dam_type",
    "certain_mask",
    "precision",
    "confidence",
    "resolution",
    "z_level",
}
COORDINATE_FILENAME = re.compile(r"[-+]?\d{1,3}\.\d+[_,-][-+]?\d{1,2}\.\d+")
CJK_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
WINDOWS_ABSOLUTE = re.compile(r"\b[A-Za-z]:[\\/]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    return parser.parse_args()


def is_machine_absolute_path(text: str) -> bool:
    slash = chr(47)
    markers = tuple(slash + name + slash for name in ("root", "home", "Users", "mnt", "tmp"))
    return any(marker in text for marker in markers) or bool(WINDOWS_ABSOLUTE.search(text))


def iter_release_files(root: Path):
    ignored_parts = {".git", ".venv", "venv", "__pycache__", "outputs"}
    for path in root.rglob("*"):
        if any(part in ignored_parts for part in path.parts):
            continue
        if path.is_symlink() or path.is_file():
            yield path


def audit_text(path: Path, errors: list[str]) -> None:
    if path.suffix not in TEXT_SUFFIXES:
        return
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        errors.append(f"Text file is not valid UTF-8: {path}")
        return
    if is_machine_absolute_path(text):
        errors.append(f"Machine-specific absolute path found: {path}")
    if path.suffix in CODE_SUFFIXES and CJK_CHARACTER.search(text):
        errors.append(f"CJK text found in public source code: {path}")


def audit_csv(path: Path, root: Path, errors: list[str]) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            exact_identifiers = find_exact_identifiers(headers)
            if exact_identifiers:
                errors.append(f"Exact identifiers in {path}: {exact_identifiers}")

            relative = path.relative_to(root)
            normalized_headers = {header.strip().lower() for header in headers}
            if relative.parts and relative.parts[0] == "anonymized_segment_dof":
                forbidden = sorted(normalized_headers.intersection(STRICT_SEGMENT_FORBIDDEN))
                if forbidden:
                    errors.append(f"Forbidden segment fields in {path}: {forbidden}")

            count_columns = infer_count_columns_from_headers(headers)
            for row_number, row in enumerate(reader, start=2):
                for column in count_columns:
                    if is_primary_small_count(row.get(column), threshold=4):
                        errors.append(
                            f"Unsuppressed count from one to four in {path}:{row_number}:{column}"
                        )
    except (csv.Error, UnicodeDecodeError) as exc:
        errors.append(f"Cannot parse CSV {path}: {exc}")


def infer_count_columns_from_headers(headers: list[str]) -> list[str]:
    empty = {header: [] for header in headers}
    import pandas as pd

    return infer_count_columns(pd.DataFrame(empty))


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    files = list(iter_release_files(root))

    for path in files:
        if path.is_symlink():
            errors.append(f"Symbolic links are not allowed in the public package: {path}")
            continue
        if path.suffix.lower() in PROHIBITED_SUFFIXES:
            errors.append(f"Prohibited file type: {path}")
        if COORDINATE_FILENAME.search(path.name):
            errors.append(f"Coordinate-like filename: {path}")
        audit_text(path, errors)
        if path.suffix.lower() == ".csv":
            audit_csv(path, root, errors)

    if errors:
        print("PUBLIC RELEASE AUDIT FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"PUBLIC RELEASE AUDIT PASSED: {len(files)} files checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
