from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def build_user_prompt(entry: Dict[str, Any], *, include_domain: bool = True) -> str:
    parts: List[str] = []
    tags = entry.get("tags", {}) or {}
    topics = tags.get("topics", []) or []

    if include_domain and topics:
        parts.append(f"Domain: {' > '.join(topics)}")
        parts.append("")

    parts.append(entry.get("question", ""))

    return "\n".join(parts).strip()


def build_plain_prompts(entry: Dict[str, Any]) -> Tuple[str, str]:
    system_prompt = entry.get("system_prompt", "")
    user_prompt = entry.get("question", "")
    return system_prompt, user_prompt


def apply_memory(user_prompt: str, memory: Optional[str]) -> str:
    if not memory:
        return user_prompt
    return (
        "You have the following memories from prior training. Use them if relevant:\n"
        f"{memory}\n\nTask:\n{user_prompt}"
    )


def load_memory_context(path: Path) -> Dict[str, str]:
    if path.suffix.lower() == ".jsonl":
        items: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return {
            str(item["task_id"]): str(item["memory"])
            for item in items
            if "task_id" in item and "memory" in item
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.items()}


def build_prompts(
    entry: Dict[str, Any],
    *,
    include_system_prompt: bool = True,
    include_domain: bool = True,
) -> Tuple[str, str]:
    system_prompt = entry.get("system_prompt", "") if include_system_prompt else ""
    user_prompt = build_user_prompt(entry, include_domain=include_domain)
    return system_prompt, user_prompt
