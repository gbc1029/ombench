from __future__ import annotations

import json
import logging
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


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

    def solve(self, payload: Dict[str, Any],timeout = self.timeout, *, dry_run: bool = False) -> Dict[str, Any]:
        if dry_run:
            return {"status": "dry_run", "request": payload}

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint.url(),
            data=data,
            headers=self.headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
        return json.loads(body)

    def solve_with_retry(
        self,
        payload: Dict[str, Any],
        *,
        dry_run: bool = False,
        max_retries: int = 1,
        backoff_base: float = 2.0,
    ) -> Dict[str, Any]:
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            timeout = self.timeout if attempt == 0 else 2 * self.timeout
            try:
                return self.solve(payload,timeout = timeout, dry_run=dry_run)
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = backoff_base ** attempt
                    logger.warning(
                        "solve attempt %d/%d failed (%s), retrying in %.1fs",
                        attempt + 1, max_retries + 1, exc, wait,
                    )
                    time.sleep(wait)
        raise last_exc  # type: ignore[misc]
