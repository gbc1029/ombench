from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

YES_VALUES = {"是", "Yes", "yes", "Y", "YES", "true", "True", "命中"}

MAX_RETRIES = 2


def _clean_and_parse_json(json_str: str) -> Any:
    if not json_str:
        raise ValueError("Empty JSON string")

    json_str = json_str.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(r",\s*\]", "]", json_str)
    fixed = re.sub(r",\s*\}", "}", fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    quoted = json_str.replace("\u201c", '"').replace("\u201d", '"')
    try:
        return json.loads(quoted)
    except json.JSONDecodeError:
        pass

    quoted_fixed = re.sub(r",\s*\]", "]", quoted)
    quoted_fixed = re.sub(r",\s*\}", "}", quoted_fixed)
    try:
        return json.loads(quoted_fixed)
    except json.JSONDecodeError:
        pass

    if json_str.startswith("[") and not json_str.endswith("]"):
        try:
            return json.loads(json_str + "]")
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not parse JSON")


def _extract_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
    if not text:
        return None

    try:
        parsed = _clean_and_parse_json(text)
        if isinstance(parsed, list):
            return parsed
    except ValueError:
        pass

    patterns = [
        r"```json\s*(\[[\s\S]*\])\s*```",
        r"```\s*(\[[\s\S]*\])\s*```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                parsed = _clean_and_parse_json(match.group(1))
                if isinstance(parsed, list):
                    return parsed
            except ValueError:
                continue

    start_positions = [m.start() for m in re.finditer(r"\[", text)]
    for start in start_positions:
        bracket_count = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            char = text[i]
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if not in_string:
                if char == "[":
                    bracket_count += 1
                elif char == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        candidate = text[start:i + 1]
                        try:
                            parsed = _clean_and_parse_json(candidate)
                            if isinstance(parsed, list):
                                return parsed
                        except ValueError:
                            pass
                        break

    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        candidate = text[first_bracket:last_bracket + 1]
        try:
            parsed = _clean_and_parse_json(candidate)
            if isinstance(parsed, list):
                return parsed
        except ValueError:
            pass

    return None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def parse_rubric_array(
    rubric_array: List[Dict[str, Any]],
    rubrics: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}

    for item in rubric_array:
        rubric_id = item.get("rubric_id")
        if rubric_id is None:
            rubric_id = item.get("rubric_number")
        if rubric_id is None:
            continue
        try:
            rubric_id = int(rubric_id)
        except (ValueError, TypeError):
            continue

        status = str(item.get("status", "")).strip()
        justification = str(item.get("justification", item.get("reason", ""))).strip()

        met_field = item.get("met")
        if met_field is not None:
            binary_score = 1 if met_field is True or str(met_field).lower() in ("true", "1") else 0
        else:
            binary_score = 1 if status in YES_VALUES else 0

        results[rubric_id] = {
            "status": status,
            "binary_score": binary_score,
            "justification": justification,
        }

    for rubric in rubrics:
        rn_raw = rubric.get("rubric_number")
        if rn_raw is None:
            continue
        try:
            rn = int(rn_raw)
        except (ValueError, TypeError):
            continue
        if rn not in results:
            results[rn] = {
                "status": "否",
                "binary_score": "NA",
                "justification": "解析失败",
            }

    return results


def convert_scores(
    raw_results: Dict[int, Dict[str, Any]],
    rubrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    score = 0
    max_score = 0
    has_valid = False
    rubric_results: List[Dict[str, Any]] = []

    for rubric in rubrics:
        rn_raw = rubric.get("rubric_number")
        weight = rubric.get("rubric_weight", 0)
        if rn_raw is None:
            continue
        try:
            rubric_num: int = int(rn_raw)
        except (ValueError, TypeError):
            continue

        if weight > 0:
            max_score += weight

        raw = raw_results.get(rubric_num, {"binary_score": 0})
        bs = raw["binary_score"]

        if bs == "NA":
            rubric_results.append({
                "rubric_number": rubric_num,
                "weight": weight,
                "met": None,
                "reason": raw.get("justification", "解析失败"),
            })
            continue

        has_valid = True
        met = bs == 1
        if met:
            score += weight

        rubric_results.append({
            "rubric_number": rubric_num,
            "weight": weight,
            "met": met,
            "reason": raw.get("justification", ""),
        })

    if not has_valid:
        max_score = 0

    return {
        "rubric_results": rubric_results,
        "score": score,
        "max_score": max_score,
    }


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
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "max_tokens": self.settings.max_tokens,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        if self.dry_run:
            return {"choices": [{"message": {"content": "[]"}}]}

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
        last_error: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._call_api(system_prompt, user_prompt)
                answer_text = _extract_answer_text(response)
                if not answer_text:
                    last_error = ValueError("Empty answer from judge")
                    logger.warning("Judge returned empty answer (attempt %d/%d)", attempt + 1, MAX_RETRIES)
                    continue

                parsed_array = _extract_json_array(answer_text)
                if parsed_array is not None:
                    return {"rubric_array": parsed_array}

                parsed_obj = _extract_json_object(answer_text)
                if parsed_obj is not None:
                    arr = parsed_obj.get("rubric_results")
                    if isinstance(arr, list):
                        return {"rubric_array": arr}
                    return {"rubric_array": [], "raw": answer_text}

                last_error = ValueError(f"Could not parse judge response as JSON array: {answer_text[:200]}")
                logger.warning("JSON parse failed (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, answer_text[:200])

            except Exception as exc:
                last_error = exc
                logger.warning("Judge API call failed (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, exc)

        return {"rubric_array": [], "raw": str(last_error) if last_error else "unknown error"}

    def judge_batch(self, system_prompt: str, user_prompt: str, expected_count: int) -> List[Dict[str, Any]]:
        if expected_count == 1:
            return [self.judge(system_prompt, user_prompt)]

        last_error: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._call_api(system_prompt, user_prompt)
                answer_text = _extract_answer_text(response)

                if not answer_text:
                    last_error = ValueError("Empty answer from judge")
                    logger.warning("Judge batch returned empty answer (attempt %d/%d)", attempt + 1, MAX_RETRIES)
                    continue

                parsed_array = _extract_json_array(answer_text)
                if parsed_array is not None:
                    if len(parsed_array) == expected_count:
                        all_inner = all(isinstance(item, list) for item in parsed_array)
                        if all_inner:
                            return [{"rubric_array": inner} for inner in parsed_array]

                    if len(parsed_array) > 0 and isinstance(parsed_array[0], dict):
                        return [{"rubric_array": parsed_array}] + [
                            {"rubric_array": [], "raw": "missing"} for _ in range(expected_count - 1)
                        ]

                last_error = ValueError(f"Batch parse failed: {answer_text[:200]}")
                logger.warning("Batch JSON parse failed (attempt %d/%d)", attempt + 1, MAX_RETRIES)

            except Exception as exc:
                last_error = exc
                logger.warning("Judge batch API call failed (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, exc)

        return [{"rubric_array": [], "raw": str(last_error) if last_error else "unknown error"}] + [
            {"rubric_array": [], "raw": "missing"} for _ in range(expected_count - 1)
        ]


# Keep backward-compatible aliases
SolveJudge = OpenAIJudge


class CallableJudge(BaseJudge):
    def __init__(self, fn) -> None:
        self.fn = fn

    def judge(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return self.fn(system_prompt, user_prompt)
