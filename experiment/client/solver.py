from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SolveEndpoint:
    base_url: str
    endpoint: str = "/task/solve"

    def url(self) -> str:
        return f"{self.base_url.rstrip('/')}{self.endpoint}"


class SolveClient:
    def __init__(
        self,
        base_url: str,
        endpoint: str = "/task/solve",
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.endpoint = SolveEndpoint(base_url=base_url, endpoint=endpoint)
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout

    def solve(self, payload: Dict[str, Any], *, dry_run: bool = False) -> Dict[str, Any]:
        if dry_run:
            return {"status": "dry_run", "request": payload}

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint.url(),
            data=data,
            headers=self.headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = response.read().decode("utf-8")
        return json.loads(body)
