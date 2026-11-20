#!/usr/bin/env python3
"""Run the existing split and macro-F1 code on invented CPU-only data."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.paired_bootstrap import _macro_f1

SPEC = importlib.util.spec_from_file_location(
    "temporal_split", ROOT / "scripts" / "06_temporal_split.py"
)
assert SPEC is not None and SPEC.loader is not None
temporal_split = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(temporal_split)

TIMES = [*range(0, 1400, 100), 1330, 1400, 1500, 1530, 1600, 1700]
PARTITIONS = ("train", "val", "test", "gap_excluded")
PREDICTIONS = {"stub_a": [0, 0], "stub_b": [0, 1]}


def make_rows():
    return [
        {"gravityspy_id": f"toy-{i:02d}", "event_time": t, "truth": i % 2}
        for i, t in enumerate(TIMES)
    ]


def canonical_json(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def audit_memberships(parts, rows, gap):
    by_id = {row["gravityspy_id"]: row["event_time"] for row in rows}
    members = [item for part in PARTITIONS for item in parts[part]]
    if len(members) != len(set(members)):
        raise ValueError("an event belongs to multiple split positions")
    if set(members) != set(by_id):
        raise ValueError("split membership does not account for every input event")
    for left, right in zip(PARTITIONS[:2], PARTITIONS[1:3]):
        if not parts[left] or not parts[right]:
            raise ValueError("a scoring partition is empty")
        delta = min(by_id[i] for i in parts[right]) - max(by_id[i] for i in parts[left])
        if delta < gap:
            raise ValueError(f"{left}/{right} violates the temporal gap")


def run(rows):
    rows = sorted(rows, key=lambda row: row["gravityspy_id"])
    if len({row["gravityspy_id"] for row in rows}) != len(rows):
        raise ValueError("input IDs must be unique")
    frame = pd.DataFrame(rows)
    boundaries = temporal_split.compute_split_boundaries(frame["event_time"].to_numpy())
    assigned = temporal_split.assign_splits(frame, boundaries)
    parts = {
        part: assigned.loc[assigned["split"] == part, "gravityspy_id"].tolist()
        for part in PARTITIONS
    }
    audit_memberships(parts, rows, temporal_split.GAP_SECONDS)
    tests = assigned.loc[assigned["split"] == "test"].sort_values("event_time")
    truth = tests["truth"].to_numpy()
    scores = {
        label: round(_macro_f1(truth, prediction), 9)
        for label, prediction in PREDICTIONS.items()
    }
    payload = {"rows": rows, "predictions": PREDICTIONS}
    return {
        "schema_version": 1,
        "scope": "synthetic CPU smoke only; not a scientific result",
        "inputs_sha256": hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
        "minimum_gap_seconds": temporal_split.GAP_SECONDS,
        "boundaries": boundaries,
        "memberships": parts,
        "prediction_order": tests["gravityspy_id"].tolist(),
        "macro_f1": scores,
        "bootstrap_run": False,
        "training_seeds_run": False,
    }


if __name__ == "__main__":
    sys.stdout.write(canonical_json(run(make_rows())))
