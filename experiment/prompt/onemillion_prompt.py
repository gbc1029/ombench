from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


def apply_memory(user_prompt: str, memory: Union[str, List[Dict[str, Any]], None]) -> str:
    if not memory:
        return user_prompt
        
    if isinstance(memory, str):
        return (
            "You have the following memories from prior training. Use them if relevant:\n"
            f"{memory}\n\nTask:\n{user_prompt}"
        )
        
    if isinstance(memory, list):
        high = []
        mid = []
        low = []
        
        for mem in memory:
            if not isinstance(mem, dict):
                continue
                
            metadata = mem.get("metadata", {})
            bucket = None
            
            if isinstance(metadata, dict):
                bucket = metadata.get("quality_bucket")
                if bucket is None:
                    model_extra = metadata.get("model_extra", {})
                    if isinstance(model_extra, dict):
                        bucket = model_extra.get("quality_bucket")
            else:
                if hasattr(metadata, "model_extra") and metadata.model_extra:
                    bucket = metadata.model_extra.get("quality_bucket")
                if bucket is None and hasattr(metadata, "quality_bucket"):
                    bucket = getattr(metadata, "quality_bucket")
                    
            if bucket not in ("high", "mid", "low"):
                success = None
                if isinstance(metadata, dict):
                    success = metadata.get("success")
                elif hasattr(metadata, "success"):
                    success = getattr(metadata, "success")
                    
                if success is None:
                    success = mem.get("success")
                    
                if success is True:
                    bucket = "high"
                elif success is False:
                    bucket = "low"
                else:
                    bucket = "mid"
                    
            content = mem.get("memory", mem.get("content", ""))
            if not content:
                continue
                
            if bucket == "high":
                high.append(str(content))
            elif bucket == "low":
                low.append(str(content))
            else:
                mid.append(str(content))
                
        parts = ["You have the following retrieved memories to guide your answer:"]
        if high:
            parts.append("\n\n--- HIGH-SCORE EXAMPLES (Excellent approaches to follow) ---\n" + "\n\n".join(high))
        if mid:
            parts.append("\n\n--- MID-SCORE EXAMPLES (Useful but may have flaws, use with caution) ---\n" + "\n\n".join(mid))
        if low:
            parts.append("\n\n--- LOW-SCORE FAILURES (Approaches that failed, avoid these mistakes) ---\n" + "\n\n".join(low))
            
        prefix = "".join(parts)
        return f"{prefix}\n\nTask:\n{user_prompt}"
        
    return user_prompt


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
