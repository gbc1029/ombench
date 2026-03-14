from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from experiment.client.solver import SolveClient
from experiment.config.settings import SolveSettings
from memrl.providers.base import BaseLLM, LLMError


def _messages_to_prompts(messages: List[Dict[str, str]]) -> Tuple[str, str]:
    system_parts: List[str] = []
    user_parts: List[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if role == "system":
            system_parts.append(content)
        else:
            user_parts.append(f"{role}: {content}" if role else content)
    return "\n\n".join(system_parts).strip(), "\n\n".join(user_parts).strip()


@dataclass
class SolveOverrides:
    request_id: Optional[str] = None
    model: Optional[str] = None
    timeout: Optional[int] = None
    step_limit: Optional[int] = None
    include_task_prompt: Optional[bool] = None


class SolveLLM(BaseLLM):
    def __init__(
        self,
        client: SolveClient,
        settings: SolveSettings,
        *,
        default_task_id: Optional[str] = None,
        dry_run: bool = False,
    ) -> None:
        super().__init__()
        self.client = client
        self.settings = settings
        self.default_task_id = default_task_id
        self.dry_run = dry_run

    def solve_raw(
        self,
        messages: List[Dict[str, str]],
        *,
        task_id: Optional[str] = None,
        overrides: Optional[SolveOverrides] = None,
    ) -> Dict[str, Any]:
        system_prompt, user_prompt = _messages_to_prompts(messages)
        override_payload: Dict[str, Any] = {
            "task_id": task_id or self.default_task_id or self.settings.extra.get("task_id"),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }
        if overrides:
            for key in ("request_id", "model", "timeout", "step_limit", "include_task_prompt"):
                value = getattr(overrides, key)
                if value is not None:
                    override_payload[key] = value

        payload = self.settings.build_payload(overrides=override_payload)
        if not payload.get("task_id"):
            raise LLMError("task_id is required for SolveLLM requests")
        return self.client.solve(payload, dry_run=self.dry_run)

    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        overrides = SolveOverrides(
            request_id=kwargs.get("request_id"),
            model=kwargs.get("model"),
            timeout=kwargs.get("timeout"),
            step_limit=kwargs.get("step_limit"),
            include_task_prompt=kwargs.get("include_task_prompt"),
        )
        response = self.solve_raw(
            messages,
            task_id=kwargs.get("task_id"),
            overrides=overrides,
        )
        if not isinstance(response, dict):
            raise LLMError("Unexpected response type from solver")
        result = response.get("result")
        if isinstance(result, dict):
            answer = result.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer
        return json.dumps(response, ensure_ascii=False)

    def extract_keywords(self, text: str, max_keywords: int = 8) -> List[str]:
        prompt = (
            f"Extract up to {max_keywords} key concepts or keywords from the following text. "
            "Return only the keywords separated by commas.\n\n"
            f"Text: {text}\n\nKeywords:"
        )
        message = [{"role": "user", "content": prompt}]
        response = self.generate(message)
        keywords = []
        for keyword in response.split(","):
            keyword = keyword.strip().strip("\"' ")
            if keyword:
                keywords.append(keyword.lower())
        return keywords[:max_keywords]

    def generate_script(self, trajectory: str) -> str:
        prompt = (
            "Analyze the following task trajectory and create a concise, high-level script with 3-5 steps.\n\n"
            f"Trajectory:\n{trajectory}\n\nHigh-level script:"
        )
        message = [{"role": "user", "content": prompt}]
        return self.generate(message)
