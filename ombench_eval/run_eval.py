from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from bridge.dataset.onemillion_loader import load_entries
from experiment.client.solver import SolveClient
from experiment.config.settings import SolveSettings, resolve_model_alias
from experiment.prompt.onemillion_prompt import (
    apply_memory,
    build_plain_prompts,
    load_memory_context,
)
from ombench_eval.evaluator import score_response
from ombench_eval.judge import JudgeSettings, OpenAIJudge


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _build_log_path(log_dir: Path, log_file: Optional[Path]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    if log_file is None:
        filename = f"run_eval_{time.strftime('%Y%m%d_%H%M%S')}.log"
        return log_dir / filename
    log_file = Path(log_file)
    if log_file.is_absolute():
        return log_file
    return log_dir / log_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OMBench evaluation for Synergy outputs")
    parser.add_argument("--mode", choices=["rubric", "plain", "memrl"], default="plain")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "OneMillion-Bench",
    )
    parser.add_argument("--responses-file", type=Path, default=None)
    parser.add_argument("--memory-context", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/results.jsonl"))
    parser.add_argument("--base-url", default="http://10.245.198.39:8000")
    parser.add_argument("--endpoint", default="/task/solve")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--judge-model", default="qwen3.5-397b-a17b")
    parser.add_argument(
        "--judge-base-url",
        default="https://holos.openapi-qb.sii.edu.cn",
        help="Base URL for judge API",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Timeout for generation requests (default: 1200)",
    )
    parser.add_argument(
        "--judge-timeout",
        type=int,
        default=600,
        help="Timeout for scoring/judge requests (default: 600)",
    )
    parser.add_argument(
        "--judge-retries",
        type=int,
        default=2,
        help="Max attempts for each judge scoring call (default: 2)",
    )
    parser.add_argument("--step-limit", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _generate_response(
    client: SolveClient,
    settings: SolveSettings,
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    payload = settings.build_payload(
        overrides={"system_prompt": system_prompt, "user_prompt": user_prompt}
    )
    return client.solve(payload, dry_run=settings.extra.get("dry_run", False))


def main() -> None:
    args = parse_args()
    entries = load_entries(args.dataset_dir)
    memory_context = load_memory_context(args.memory_context) if args.memory_context else {}

    log_path = _build_log_path(args.log_dir, args.log_file)
    handlers = [logging.FileHandler(log_path), logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    logging.getLogger(__name__).info("Logging to %s", log_path)

    judge = OpenAIJudge(
        JudgeSettings(
            base_url=args.judge_base_url,
            model=args.judge_model,
            timeout=args.judge_timeout,
            max_retries=args.judge_retries,
        ),
        dry_run=args.dry_run,
    )

    client = SolveClient(base_url=args.base_url, endpoint=args.endpoint, timeout=args.timeout)
    settings = SolveSettings(
        base_url=args.base_url,
        endpoint=args.endpoint,
        benchmark="onemillion",
        model=resolve_model_alias(args.model),
        timeout=args.timeout,
        step_limit=args.step_limit,
        request_id="omb-gen",
        system_prompt="",
        user_prompt="",
        extra={"dry_run": args.dry_run},
    )

    results: List[Dict[str, Any]] = []

    if args.mode == "rubric":
        if not args.responses_file:
            raise SystemExit("--responses-file is required for rubric mode")
        responses = _load_jsonl(args.responses_file)
        for idx, item in enumerate(responses, start=1):
            if args.limit and idx > args.limit:
                break
            task_id = str(item.get("task_id"))
            answer = item.get("answer", "")
            entry = entries.get(task_id)
            if not entry:
                continue
            score = score_response(
                judge=judge,
                question=entry.get("question", ""),
                response=str(answer),
                rubrics=entry.get("rubrics", []),
                system_prompt=entry.get("system_prompt"),
            )
            results.append({"task_id": task_id, "score": score})

    else:
        for idx, (task_id, entry) in enumerate(entries.items(), start=1):
            if args.limit and idx > args.limit:
                break
            system_prompt, user_prompt = build_plain_prompts(entry)
            if args.mode == "memrl":
                user_prompt = apply_memory(user_prompt, memory_context.get(task_id))
            response = _generate_response(client, settings, system_prompt, user_prompt)
            answer = response.get("result", {}).get("answer", "") if isinstance(response, dict) else ""
            score = score_response(
                judge=judge,
                question=entry.get("question", ""),
                response=str(answer),
                rubrics=entry.get("rubrics", []),
                system_prompt=entry.get("system_prompt"),
            )
            results.append({"task_id": task_id, "response": answer, "score": score})

    if args.output:
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
