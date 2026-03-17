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

    def solve(
        self,
        payload: Dict[str, Any],
        timeout: Optional[int] = None,
        *,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        if timeout is None:
            timeout = self.timeout
        if dry_run:
            return {"status": "dry_run", "request": payload}

        task_id = payload.get("task_id", "?")
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint.url(),
            data=data,
            headers=self.headers,
            method="POST",
        )
        t0 = time.monotonic()
        send_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info("[GEN] task=%s send=%s timeout=%s", task_id, send_ts, timeout)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
            elapsed = time.monotonic() - t0
            recv_ts = time.strftime("%Y-%m-%d %H:%M:%S")
            logger.info("[GEN] task=%s recv=%s elapsed=%.1fs", task_id, recv_ts, elapsed)
            return json.loads(body)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            recv_ts = time.strftime("%Y-%m-%d %H:%M:%S")
            logger.info("[GEN] task=%s fail=%s elapsed=%.1fs timeout=%s error=%s",
                        task_id, recv_ts, elapsed, timeout, exc)
            raise

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
            base_timeout = self.timeout if self.timeout is not None else None
            timeout = base_timeout if attempt == 0 else (2 * base_timeout if base_timeout is not None else None)
            try:
                return self.solve(payload, timeout=timeout, dry_run=dry_run)
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
