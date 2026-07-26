"""Shared result readers for dense degradation plots and reports."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


V2_PROVENANCE_FIELDS = {
    "representation",
    "checkpoint",
    "checkpoint_sha256",
    "probe_config",
    "dataset_identity",
    "source_commit",
    "source_dirty",
}


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


def read_voc_results(
    path: str | Path | None,
    *,
    as_rows: bool = False,
    expected_metric_version: str | None = None,
    expected_probe_seed: int | None = None,
    expected_checkpoint_key: str | None = None,
    require_provenance: bool = False,
):
    """Read VOC results, optionally enforcing an explicit evaluation protocol."""
    if not path or not Path(path).is_file():
        if require_provenance or any(
            value is not None
            for value in (
                expected_metric_version,
                expected_probe_seed,
                expected_checkpoint_key,
            )
        ):
            raise FileNotFoundError(f"Required VOC results file not found: {path}")
        return [] if as_rows else {}
    with Path(path).open() as handle:
        rows = sorted(json.load(handle), key=lambda row: row["epoch"])
    if not rows and (
        require_provenance
        or any(
            value is not None
            for value in (
                expected_metric_version,
                expected_probe_seed,
                expected_checkpoint_key,
            )
        )
    ):
        raise ValueError(f"Required VOC results file is empty: {path}")
    expectations = {
        "metric_version": expected_metric_version,
        "probe_seed": expected_probe_seed,
        "checkpoint_key": expected_checkpoint_key,
    }
    for field, expected in expectations.items():
        if expected is None:
            continue
        for row in rows:
            if field not in row:
                raise ValueError(
                    f"VOC result for epoch {row.get('epoch')} is missing required "
                    f"protocol field '{field}'"
                )
            if row[field] != expected:
                raise ValueError(
                    f"VOC result for epoch {row.get('epoch')} has {field}="
                    f"{row[field]!r}; expected {expected!r}"
                )
    if require_provenance:
        for row in rows:
            missing = sorted(V2_PROVENANCE_FIELDS.difference(row))
            if missing:
                raise ValueError(
                    f"VOC result for epoch {row.get('epoch')} is missing required "
                    f"provenance fields: {missing}"
                )
            if not re.fullmatch(r"[0-9a-f]{64}", str(row["checkpoint_sha256"])):
                raise ValueError(
                    f"VOC result for epoch {row.get('epoch')} has invalid "
                    "checkpoint_sha256"
                )
            if not re.fullmatch(r"[0-9a-f]{40,64}", str(row["source_commit"])):
                raise ValueError(
                    f"VOC result for epoch {row.get('epoch')} has invalid "
                    "source_commit"
                )
            if not isinstance(row["probe_config"], dict):
                raise ValueError(
                    f"VOC result for epoch {row.get('epoch')} has invalid "
                    "probe_config"
                )
            if not isinstance(row["dataset_identity"], dict):
                raise ValueError(
                    f"VOC result for epoch {row.get('epoch')} has invalid "
                    "dataset_identity"
                )
            if not isinstance(row["source_dirty"], bool):
                raise ValueError(
                    f"VOC result for epoch {row.get('epoch')} has invalid "
                    "source_dirty"
                )
            if row["source_dirty"]:
                raise ValueError(
                    f"VOC result for epoch {row.get('epoch')} was produced "
                    "from a dirty source tree"
                )
            expected_representation = (
                "ema_teacher"
                if row.get("checkpoint_key") == "teacher"
                else "student"
                if row.get("checkpoint_key") == "student"
                else None
            )
            if row["representation"] != expected_representation:
                raise ValueError(
                    f"VOC result for epoch {row.get('epoch')} has representation="
                    f"{row['representation']!r}, inconsistent with checkpoint_key="
                    f"{row.get('checkpoint_key')!r}"
                )
        epochs = [int(row["epoch"]) for row in rows]
        if len(epochs) != len(set(epochs)):
            raise ValueError("VOC results contain duplicate checkpoint epochs")
        for field in (
            "source_commit",
            "source_dirty",
            "probe_config",
            "dataset_identity",
        ):
            values = {
                json.dumps(row[field], sort_keys=True)
                for row in rows
            }
            if len(values) != 1:
                raise ValueError(f"VOC results mix {field} values")
    if as_rows:
        return rows
    return {int(row["epoch"]): float(row["miou"]) for row in rows}
