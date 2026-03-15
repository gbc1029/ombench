from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ombench_eval.prompts import (
    RUBRIC_JUDGE_SYSTEM_PROMPT,
    build_batch_rubric_judge_prompt,
    build_rubric_judge_prompt,
)
from ombench_eval.judge import BaseJudge

logger = logging.getLogger(__name__)


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    message = str(exc).lower()
    return "timeout" in message or "timed out" in message


def _fallback_score(rubrics: List[Dict[str, Any]], results: Dict[str, Any]) -> Dict[str, Any]:
    rubric_results = results.get("rubric_results", []) if isinstance(results, dict) else []
    if not isinstance(rubric_results, list):
        rubric_results = []
    weight_by_number = {r.get("rubric_number"): r.get("rubric_weight") for r in rubrics}
    score = 0
    max_score = 0
    for rubric in rubrics:
        weight = rubric.get("rubric_weight", 0)
        if weight > 0:
            max_score += weight
    for item in rubric_results:
        number = item.get("rubric_number")
        met = item.get("met")
        weight = item.get("weight")
        if weight is None:
            weight = weight_by_number.get(number, 0)
        if met is True:
            score += weight or 0
    return {"rubric_results": rubric_results, "score": score, "max_score": max_score}


def score_response(
    *,
    judge: BaseJudge,
    question: str,
    response: str,
    rubrics: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    user_prompt = build_rubric_judge_prompt(
        question=question,
        response=response,
        rubrics=rubrics,
        system_prompt=system_prompt,
    )
    result = judge.judge(RUBRIC_JUDGE_SYSTEM_PROMPT, user_prompt)
    if not isinstance(result, dict):
        return {"raw": result}
    if "score" in result and "max_score" in result:
        return result
    return _fallback_score(rubrics, result)


def _score_batch_group(
    judge: BaseJudge,
    group: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if len(group) == 1:
        item = group[0]
        return [score_response(
            judge=judge,
            question=item.get("question", ""),
            response=item.get("response", ""),
            rubrics=item.get("rubrics", []),
            system_prompt=item.get("system_prompt"),
        )]

    user_prompt = build_batch_rubric_judge_prompt(items=group)
    raw_results = judge.judge_batch(RUBRIC_JUDGE_SYSTEM_PROMPT, user_prompt, expected_count=len(group))

    scored: List[Dict[str, Any]] = []
    for idx, raw in enumerate(raw_results):
        if not isinstance(raw, dict):
            scored.append({"raw": raw})
            continue
        if "score" in raw and "max_score" in raw:
            scored.append(raw)
        else:
            rubrics = group[idx].get("rubrics", []) if idx < len(group) else []
            scored.append(_fallback_score(rubrics, raw))
    return scored


def score_responses_batch(
    *,
    judge: BaseJudge,
    items: List[Dict[str, Any]],
    batch_size: int = 1,
    max_workers: int = 4,
    show_progress: bool = False,
) -> List[Dict[str, Any]]:

    batch_size = max(batch_size, 1)
    max_workers = max(max_workers, 1)

    groups: List[List[Dict[str, Any]]] = []
    for i in range(0, len(items), batch_size):
        groups.append(items[i : i + batch_size])

    all_results: List[Optional[List[Dict[str, Any]]]] = [None] * len(groups)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(_score_batch_group, judge, group): gidx
            for gidx, group in enumerate(groups)
        }
        progress = (
            tqdm(total=len(future_map), desc="Score", unit="batch")
            if show_progress
            else None
        )
        try:
            for future in as_completed(future_map):
                gidx = future_map[future]
                try:
                    all_results[gidx] = future.result()
                except Exception as exc:
                    logger.error("Score batch group %d failed: %s", gidx, exc)
                    group = groups[gidx]
                    if _is_timeout_error(exc):
                        task_ids = [
                            str(item.get("task_id"))
                            for item in group
                            if item.get("task_id")
                        ]
                        if task_ids:
                            logger.warning(
                                "Score timeout for task_ids: %s",
                                ", ".join(task_ids),
                            )
                        else:
                            logger.warning("Score timeout for task_ids: unknown")
                    all_results[gidx] = [{"raw": str(exc)} for _ in group]
                finally:
                    if progress is not None:
                        progress.update(1)
        finally:
            if progress is not None:
                progress.close()

    flat: List[Dict[str, Any]] = []
    for group_result in all_results:
        if group_result is not None:
            flat.extend(group_result)
        else:
            flat.append({"raw": "missing"})
    return flat
