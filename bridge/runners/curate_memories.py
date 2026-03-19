"""Offline CLI script to deduplicate an existing memory pool checkpoint.

Loads ``cube/textual_memory.json`` from a checkpoint snapshot, removes
near-duplicate memories based on cosine similarity of their embedding
vectors, and writes the curated result back (or to a separate path).

Usage::

    python -m bridge.runners.curate_memories \
        --checkpoint checkpoints/batch_pipeline/snapshot/20260319_205903 \
        --novelty-threshold 0.95 \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

DEFAULT_NOVELTY_THRESHOLD = 0.95
DEFAULT_REWARD_MERGE_GAIN = 0.1


# ------------------------------------------------------------------
# Core logic
# ------------------------------------------------------------------


def _load_memories(path: Path) -> list[dict[str, Any]]:
    """Read the textual_memory.json file and return the list of memory dicts."""
    _log.info("Loading memories from %s", path)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    _log.info("Loaded %d memories", len(data))
    return data


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _cosine_similarity_batch(
    kept_matrix: np.ndarray, candidate: np.ndarray
) -> np.ndarray:
    """Return cosine similarities between *candidate* and every row in *kept_matrix*.

    Parameters
    ----------
    kept_matrix:
        ``(N, D)`` array of already-kept vectors (assumed pre-normalised).
    candidate:
        ``(D,)`` vector (assumed pre-normalised).

    Returns
    -------
    np.ndarray of shape ``(N,)`` with similarity scores.
    """
    if kept_matrix.shape[0] == 0:
        return np.array([], dtype=np.float64)
    return kept_matrix @ candidate


def _normalise(vec: np.ndarray) -> np.ndarray:
    """L2-normalise a vector (return zero-vector unchanged)."""
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def curate(
    memories: list[dict[str, Any]],
    *,
    novelty_threshold: float = DEFAULT_NOVELTY_THRESHOLD,
    reward_merge_gain: float = DEFAULT_REWARD_MERGE_GAIN,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate *memories* based on cosine similarity of their vectors.

    Returns ``(kept, removed)`` lists.  The *kept* memories may have their
    ``payload.q_value`` boosted when a redundant memory is merged into them.
    """
    if not memories:
        return [], []

    sorted_memories = sorted(
        memories,
        key=lambda m: m.get("payload", {}).get("q_value", 0.0),
        reverse=True,
    )

    dim = len(sorted_memories[0].get("vector", []))
    if dim == 0:
        _log.warning("First memory has an empty vector; skipping deduplication")
        return list(memories), []

    kept: list[dict[str, Any]] = []
    kept_normed: list[np.ndarray] = []
    removed: list[dict[str, Any]] = []

    kept_matrix = np.empty((0, dim), dtype=np.float64)

    for mem in sorted_memories:
        vec = np.asarray(mem.get("vector", []), dtype=np.float64)
        if vec.shape[0] != dim:
            _log.warning(
                "Memory %s has vector dim %d (expected %d); keeping unconditionally",
                mem.get("id", "?"),
                vec.shape[0],
                dim,
            )
            kept.append(mem)
            continue

        normed = _normalise(vec)

        if kept_matrix.shape[0] == 0:
            kept.append(mem)
            kept_normed.append(normed)
            kept_matrix = normed.reshape(1, -1)
            continue

        sims = _cosine_similarity_batch(kept_matrix, normed)
        max_idx = int(np.argmax(sims))
        max_sim = float(sims[max_idx])

        if max_sim >= novelty_threshold:
            removed.append(mem)
            q_value = mem.get("payload", {}).get("q_value", 0.0)
            if q_value > 0.0:
                kept_payload = kept[max_idx].setdefault("payload", {})
                kept_payload["q_value"] = (
                    kept_payload.get("q_value", 0.0) + reward_merge_gain * q_value
                )
            _log.debug(
                "Removed %s (sim=%.4f with %s)",
                mem.get("id", "?"),
                max_sim,
                kept[max_idx].get("id", "?"),
            )
        else:
            kept.append(mem)
            kept_normed.append(normed)
            kept_matrix = np.vstack([kept_matrix, normed.reshape(1, -1)])

    return kept, removed


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------


def _print_summary(
    original: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    removed: list[dict[str, Any]],
    dry_run: bool,
) -> None:
    prefix = "[DRY-RUN] " if dry_run else ""
    print(f"\n{prefix}=== Curation Summary ===")
    print(f"  Original count : {len(original)}")
    print(f"  Kept count     : {len(kept)}")
    print(f"  Removed count  : {len(removed)}")

    def _type_counts(mems: list[dict[str, Any]]) -> Counter:
        return Counter(m.get("payload", {}).get("type", "unknown") for m in mems)

    orig_types = _type_counts(original)
    kept_types = _type_counts(kept)
    removed_types = _type_counts(removed)

    all_types = sorted(orig_types.keys())
    print("\n  Per-type breakdown:")
    print(f"  {'type':<20} {'original':>10} {'kept':>10} {'removed':>10}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10}")
    for t in all_types:
        print(
            f"  {t:<20} {orig_types[t]:>10} {kept_types.get(t, 0):>10} {removed_types.get(t, 0):>10}"
        )
    print()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deduplicate memories in a checkpoint snapshot based on cosine similarity.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint snapshot directory.",
    )
    parser.add_argument(
        "--novelty-threshold",
        type=float,
        default=DEFAULT_NOVELTY_THRESHOLD,
        help=f"Cosine similarity threshold above which memories are considered duplicates (default: {DEFAULT_NOVELTY_THRESHOLD}).",
    )
    parser.add_argument(
        "--reward-merge-gain",
        type=float,
        default=DEFAULT_REWARD_MERGE_GAIN,
        help=f"Fraction of a redundant memory's q_value to add to the kept memory (default: {DEFAULT_REWARD_MERGE_GAIN}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be removed; do not modify files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for the curated textual_memory.json (default: overwrite in-place).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    checkpoint_dir = Path(args.checkpoint)
    memory_path = checkpoint_dir / "cube" / "textual_memory.json"

    if not memory_path.exists():
        _log.error("Memory file not found: %s", memory_path)
        sys.exit(1)

    memories = _load_memories(memory_path)
    kept, removed = curate(
        memories,
        novelty_threshold=args.novelty_threshold,
        reward_merge_gain=args.reward_merge_gain,
    )
    _print_summary(memories, kept, removed, dry_run=args.dry_run)

    if args.dry_run:
        _log.info("Dry-run mode: no files were modified.")
        return

    output_path = Path(args.output) if args.output else memory_path

    if output_path == memory_path:
        backup_path = memory_path.with_suffix(".json.bak")
        _log.info("Backing up original to %s", backup_path)
        shutil.copy2(memory_path, backup_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(kept, fh, ensure_ascii=False, indent=2)

    _log.info("Wrote %d curated memories to %s", len(kept), output_path)


if __name__ == "__main__":
    main()
