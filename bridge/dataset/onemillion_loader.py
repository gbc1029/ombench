from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

SUBSETS = [
    "economics_and_finance",
    "healthcare_and_medicine",
    "industry",
    "law",
    "natural_science",
]


def load_entries(dataset_dir: Path, *, subsets: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
    dataset_dir = dataset_dir.expanduser().resolve()
    requested = set(subsets or SUBSETS)
    entries: Dict[str, Dict[str, Any]] = {}

    for subset in SUBSETS:
        if subset not in requested:
            continue
        subset_path = dataset_dir / subset / "test.json"
        if not subset_path.exists():
            continue
        with open(subset_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        for entry in rows:
            case_id = entry.get("case_id")
            lang = entry.get("language", "en")
            task_id = f"{subset}/{case_id}/{lang}"
            entries[task_id] = entry
    return entries
