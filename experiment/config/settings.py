from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


MODEL_ALIASES: Dict[str, str] = {
    "qwen": "sii-holos/Qwen 3.5 397B A17B",
    "nex": "sii-nex/Nex N1.1",
}


def resolve_model_alias(model: str) -> str:
    if not model:
        return model
    key = model.strip().lower()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    for value in MODEL_ALIASES.values():
        if model == value:
            return model
    raise ValueError(
        "--model must be one of: " + ", ".join(sorted(MODEL_ALIASES.keys()))
    )


@dataclass
class SolveSettings:
    base_url: str = "http://10.245.198.39:8000"
    endpoint: str = "/task/solve"
    benchmark: str = "onemillion"
    model: str = "sii-holos/Qwen 3.5 397B A17B"
    timeout: int = 1200
    step_limit: int = 150
    request_id: str = "omb-probe-v2"
    system_prompt: str = ""
    user_prompt: str = ""
    include_task_prompt: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def build_payload(self, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
        payload = {
            "request_id": self.request_id,
            "benchmark": self.benchmark,
            "task_id": self.extra.get("task_id"),
            "model": self.model,
            "timeout": self.timeout,
            "step_limit": self.step_limit,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "include_task_prompt": False,
        }
        payload.update(self.extra)
        if overrides:
            payload.update(overrides)
        return {k: v for k, v in payload.items() if v is not None}
