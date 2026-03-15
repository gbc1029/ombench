from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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


def _extract_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return None


def _extract_answer_text(response: Dict[str, Any]) -> str:
    """Extract answer text from OpenAI-compatible chat completion response.

    Qwen3.5 thinking mode returns content=null with the actual text in
    message.reasoning.  We try content first, then fall back to reasoning.
    """
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return ""


class BaseJudge:
    def judge(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raise NotImplementedError

    def judge_batch(self, system_prompt: str, user_prompt: str, expected_count: int) -> List[Dict[str, Any]]:
        raise NotImplementedError


@dataclass
class JudgeSettings:
    base_url: str = "https://holos.openapi-qb.sii.edu.cn"
    model: str = "qwen3.5-397b-a17b"
    api_key_env: str = "INF_API_KEY"
    timeout: int = 600
    max_tokens: int = 16384


class OpenAIJudge(BaseJudge):
    """Judge that calls an OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(self, settings: JudgeSettings, *, dry_run: bool = False) -> None:
        self.settings = settings
        self.dry_run = dry_run
        self.url = f"{settings.base_url.rstrip('/')}/v1/chat/completions"
        api_key = os.environ.get(settings.api_key_env, "")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def _call_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.settings.max_tokens,
            "temperature": 0.1,
        }
        if self.dry_run:
            return {"choices": [{"message": {"content": "{\"rubric_results\":[], \"score\":0, \"max_score\":0}"}}]}

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.settings.timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            logger.error("Judge API HTTP %d: %s", exc.code, err_body)
            raise RuntimeError(f"Judge API HTTP {exc.code}: {err_body}") from exc

    def judge(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        response = self._call_api(system_prompt, user_prompt)
        answer_text = _extract_answer_text(response)
        if not answer_text:
            return {"raw": response}
        parsed = _extract_json(answer_text)
        if parsed is not None:
            return parsed
        return {"raw": answer_text}

    def judge_batch(self, system_prompt: str, user_prompt: str, expected_count: int) -> List[Dict[str, Any]]:
        if expected_count == 1:
            return [self.judge(system_prompt, user_prompt)]

        response = self._call_api(system_prompt, user_prompt)
        answer_text = _extract_answer_text(response)

        if not answer_text:
            return [{"raw": response}] + [{"raw": "missing"} for _ in range(expected_count - 1)]

        parsed_array = _extract_json_array(answer_text)
        if parsed_array is not None and len(parsed_array) == expected_count:
            return parsed_array

        if parsed_array is not None and len(parsed_array) > 0:
            if len(parsed_array) < expected_count:
                parsed_array.extend(
                    [{"raw": "missing"} for _ in range(expected_count - len(parsed_array))]
                )
            return parsed_array[:expected_count]

        single = _extract_json(answer_text)
        if single is not None:
            return [single] + [{"raw": "missing"} for _ in range(expected_count - 1)]

        return [{"raw": answer_text}] + [{"raw": "missing"} for _ in range(expected_count - 1)]


# Keep backward-compatible aliases
SolveJudge = OpenAIJudge


class CallableJudge(BaseJudge):
    def __init__(self, fn) -> None:
        self.fn = fn

    def judge(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return self.fn(system_prompt, user_prompt)
