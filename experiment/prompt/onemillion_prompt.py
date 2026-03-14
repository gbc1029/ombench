from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


def build_user_prompt(
    entry: Dict[str, Any],
    *,
    include_domain: bool = True,
    include_rubrics: bool = True,
) -> str:
    parts: List[str] = []
    tags = entry.get("tags", {}) or {}
    topics = tags.get("topics", []) or []
    rubrics = entry.get("rubrics", []) or []

    if include_domain and topics:
        parts.append(f"Domain: {' > '.join(topics)}")
        parts.append("")

    parts.append(entry.get("question", ""))

    if include_rubrics and rubrics:
        parts.append("")
        parts.append("Your response will be evaluated on the following criteria:")
        for rubric in rubrics:
            weight = rubric.get("rubric_weight", 0)
            label = rubric.get("rubric_label", rubric.get("rubricLabel", ""))
            detail = rubric.get("rubric_detail", "")
            sign = "+" if weight > 0 else ""
            parts.append(f"  [{sign}{weight}] {label}: {detail}")

    return "\n".join(parts).strip()


def build_prompts(
    entry: Dict[str, Any],
    *,
    include_system_prompt: bool = True,
    include_domain: bool = True,
    include_rubrics: bool = True,
) -> Tuple[str, str]:
    system_prompt = entry.get("system_prompt", "") if include_system_prompt else ""
    user_prompt = build_user_prompt(
        entry,
        include_domain=include_domain,
        include_rubrics=include_rubrics,
    )
    return system_prompt, user_prompt
