from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from experiment.client.solver import SolveClient
from memrl.providers.base import LLMError


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


class BaseJudge:
    def judge(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class JudgeSettings:
    base_url: str
    endpoint: str = "/task/solve"
    model: str = "sii-holos/Qwen 3.5 397B A17B"
    timeout: int = 240
    step_limit: int = 150
    request_id: str = "omb-judge"
    task_id: str = "ombench/judge"


class SolveJudge(BaseJudge):
    def __init__(self, settings: JudgeSettings, *, dry_run: bool = False) -> None:
        self.settings = settings
        self.client = SolveClient(
            base_url=settings.base_url,
            endpoint=settings.endpoint,
            timeout=settings.timeout,
        )
        self.dry_run = dry_run

    def judge(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        payload = {
            "request_id": self.settings.request_id,
            "benchmark": "onemillion",
            "task_id": self.settings.task_id,
            "model": self.settings.model,
            "timeout": self.settings.timeout,
            "step_limit": self.settings.step_limit,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }
        response = self.client.solve(payload, dry_run=self.dry_run)
        if not isinstance(response, dict):
            raise LLMError("Judge response is not a dict")
        result = response.get("result")
        if isinstance(result, dict):
            answer = result.get("answer")
            if isinstance(answer, str):
                parsed = _extract_json(answer)
                if parsed is not None:
                    return parsed
        if isinstance(response, dict):
            text = json.dumps(response, ensure_ascii=False)
            parsed = _extract_json(text)
            if parsed is not None:
                return parsed
        return {"raw": response}


class CallableJudge(BaseJudge):
    def __init__(self, fn) -> None:
        self.fn = fn

    def judge(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return self.fn(system_prompt, user_prompt)
