from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ombench_eval.prompts import (
    RUBRIC_JUDGE_SYSTEM_PROMPT,
    build_rubric_judge_prompt,
)
from ombench_eval.judge import BaseJudge, convert_scores, parse_rubric_array

logger = logging.getLogger(__name__)


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    message = str(exc).lower()
    return "timeout" in message or "timed out" in message


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
        return {"rubric_results": [], "score": 0, "max_score": 0, "raw": result}

    raw_response = result.get("raw_response", "")
    raw_error = result.get("raw")

    rubric_array = result.get("rubric_array")
    if isinstance(rubric_array, list):
        if len(rubric_array) < len(rubrics):
            raw = raw_error or "incomplete_rubric_array"
            return {
                "rubric_results": [],
                "score": 0,
                "max_score": 0,
                "raw": raw,
                "raw_response": raw_response,
            }
        raw_results = parse_rubric_array(rubric_array, rubrics)
        scores = convert_scores(raw_results, rubrics)
        scores["raw_response"] = raw_response
        if raw_error is not None:
            scores["raw"] = raw_error
        return scores

    return {"rubric_results": [], "score": 0, "max_score": 0, "raw": result, "raw_response": raw_response}


def score_responses_batch(
    *,
    judge: BaseJudge,
    items: List[Dict[str, Any]],
    batch_size: int = 1,
    max_workers: int = 4,
    show_progress: bool = False,
) -> List[Dict[str, Any]]:

    max_workers = max(max_workers, 1)

    results: List[Optional[Dict[str, Any]]] = [None] * len(items)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(
                score_response,
                judge=judge,
                question=item.get("question", ""),
                response=item.get("response", ""),
                rubrics=item.get("rubrics", []),
                system_prompt=item.get("system_prompt"),
            ): idx
            for idx, item in enumerate(items)
        }
        progress = (
            tqdm(total=len(future_map), desc="Score", unit="task")
            if show_progress
            else None
        )
        try:
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error("Score task %d failed: %s", idx, exc)
                    results[idx] = {
                        "rubric_results": [],
                        "score": 0,
                        "max_score": 0,
                        "raw": str(exc),
                    }
                finally:
                    if progress is not None:
                        progress.update(1)
        finally:
            if progress is not None:
                progress.close()

    final: List[Dict[str, Any]] = []
    for result in results:
        if result is not None:
            final.append(result)
        else:
            final.append({"rubric_results": [], "score": 0, "max_score": 0, "raw": "missing"})
    return final
