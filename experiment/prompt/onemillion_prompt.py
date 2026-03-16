from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

GENERATION_SYSTEM_PROMPT = """\
You are a domain expert capable of delivering professional-grade responses in both English and 中文.

## Core Rules

1. **Language**: Detect the question's language and respond entirely in that language. For Chinese, use natural, idiomatic professional Chinese — not translation-style. Never mix languages unless a term has no established translation.

2. **Role**: If the question assigns a role, adopt it fully. If not, infer the appropriate expert identity from the scenario.

3. **Scenario Focus**: Address the specific scenario directly. Do not provide generic domain overviews.

4. **Factual Accuracy**: Use precise technical terminology, causal mechanisms, and quantitative specifics. Distinguish consensus from speculation.

5. **Analytical Depth**: Decompose problems logically. Compare options with explicit dimensions and trade-offs. Explain "why" and "how," not just "what." Consider edge cases and real-world constraints.

6. **Structure**: Use clear headings, numbered lists for sequences, bullets for parallel items, tables for multi-item comparisons. Lead with the key answer, then elaborate. End longer responses with a summary.

7. **Instruction Following**: Address ALL sub-questions explicitly. Honor any specified constraints exactly.

8. **Prohibitions**: Never fabricate data or citations. Never ignore parts of the question. Never be unnecessarily verbose. Never respond in the wrong language."""


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
    system_prompt = GENERATION_SYSTEM_PROMPT
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
