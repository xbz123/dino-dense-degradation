"""Shared result readers for dense degradation plots and reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def to_number(value):
    """Parse numeric CSV strings while preserving non-numeric values."""
    if value is None or value == "":
        return float("nan")
    try:
        if isinstance(value, str) and value.strip().isdigit():
            return int(value)
        return float(value)
    except (TypeError, ValueError):
        return value


def read_csv_rows(path: str | Path) -> list[dict]:
    """Read a CSV summary and sort rows by checkpoint epoch."""
    with Path(path).open() as handle:
        rows = [{key: to_number(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    return sorted(rows, key=lambda row: row["epoch"])


def read_voc_results(path: str | Path | None, *, as_rows: bool = False):
    """Read VOC mIoU results as either sorted rows or an epoch->mIoU mapping."""
    if not path or not Path(path).is_file():
        return [] if as_rows else {}
    with Path(path).open() as handle:
        rows = sorted(json.load(handle), key=lambda row: row["epoch"])
    if as_rows:
        return rows
    return {int(row["epoch"]): float(row["miou"]) for row in rows}
