from __future__ import annotations

from typing import Any, Dict, List, Optional

RUBRIC_JUDGE_SYSTEM_PROMPT = (
    "You are a strict rubric-based grader. "
    "Evaluate the model response against each rubric item and decide whether it is satisfied. "
    "Return JSON only. If unsure, set met=false."
)


def build_rubric_judge_prompt(
    *,
    question: str,
    response: str,
    rubrics: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
) -> str:
    lines: List[str] = []
    if system_prompt:
        lines.append("SYSTEM PROMPT:")
        lines.append(system_prompt)
        lines.append("")

    lines.append("QUESTION:")
    lines.append(question)
    lines.append("")

    lines.append("MODEL RESPONSE:")
    lines.append(response)
    lines.append("")

    lines.append("RUBRICS:")
    for rubric in rubrics:
        number = rubric.get("rubric_number")
        weight = rubric.get("rubric_weight")
        label = rubric.get("rubric_label", "")
        detail = rubric.get("rubric_detail", "")
        lines.append(f"- #{number} (weight {weight}) [{label}] {detail}")

    lines.append("")
    lines.append(
        "Return JSON with the following schema only (no extra text):\n"
        "{\n"
        "  \"rubric_results\": [\n"
        "    {\"rubric_number\": <int>, \"weight\": <int>, \"met\": <true|false>, \"reason\": <string>}\n"
        "  ],\n"
        "  \"score\": <int>,\n"
        "  \"max_score\": <int>\n"
        "}\n"
        "Rules: score = sum(weight for each met rubric). max_score = sum of positive weights."
    )
    return "\n".join(lines)
