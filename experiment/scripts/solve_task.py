from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment.client.solver import SolveClient
from experiment.config.settings import SolveSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a OneMillion-Bench solve request")
    parser.add_argument("--base-url", default="http://10.245.198.39:8000")
    parser.add_argument("--endpoint", default="/task/solve")
    parser.add_argument("--benchmark", default="onemillion")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", default="sii-holos/Qwen 3.5 397B A17B")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--step-limit", type=int, default=150)
    parser.add_argument("--request-id", default="omb-probe-v2")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--user-prompt", default="")
    parser.add_argument("--include-task-prompt", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Do not send the request")
    parser.add_argument("--output", type=Path, default=Path("outputs/results.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = SolveSettings(
        base_url=args.base_url,
        endpoint=args.endpoint,
        benchmark=args.benchmark,
        model=args.model,
        timeout=args.timeout,
        step_limit=args.step_limit,
        request_id=args.request_id,
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
        include_task_prompt=args.include_task_prompt if args.include_task_prompt else None,
        extra={"task_id": args.task_id},
    )
    client = SolveClient(base_url=args.base_url, endpoint=args.endpoint, timeout=args.timeout)
    payload = settings.build_payload()
    result = client.solve(payload, dry_run=args.dry_run)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
