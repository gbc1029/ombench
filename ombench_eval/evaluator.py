from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ombench_eval.prompts import RUBRIC_JUDGE_SYSTEM_PROMPT, build_rubric_judge_prompt
from ombench_eval.judge import BaseJudge


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
