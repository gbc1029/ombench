from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from bridge.dataset.onemillion_loader import load_entries
from experiment.client.solver import SolveClient
from experiment.config.settings import SolveSettings
from experiment.prompt.onemillion_prompt import build_prompts
from ombench_eval.evaluator import score_response
from ombench_eval.judge import JudgeSettings, SolveJudge


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _load_memory_context(path: Path) -> Dict[str, str]:
    if path.suffix.lower() == ".jsonl":
        items = _load_jsonl(path)
        return {str(item["task_id"]): str(item["memory"]) for item in items if "task_id" in item and "memory" in item}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.items()}


def _apply_memory(user_prompt: str, memory: Optional[str]) -> str:
    if not memory:
        return user_prompt
    return f"You have the following memories from prior training. Use them if relevant:\n{memory}\n\nTask:\n{user_prompt}"


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
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--base-url", default="http://10.245.198.154:8000")
    parser.add_argument("--endpoint", default="/task/solve")
    parser.add_argument("--model", default="holos-qwen35-397b/Qwen3.5-397B-A17B")
    parser.add_argument("--judge-model", default="holos-qwen35-397b/Qwen3.5-397B-A17B")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--step-limit", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _generate_response(client: SolveClient, settings: SolveSettings, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    payload = settings.build_payload(
        overrides={"system_prompt": system_prompt, "user_prompt": user_prompt}
    )
    return client.solve(payload, dry_run=settings.extra.get("dry_run", False))


def main() -> None:
    args = parse_args()
    entries = load_entries(args.dataset_dir)
    memory_context = _load_memory_context(args.memory_context) if args.memory_context else {}

    judge = SolveJudge(
        JudgeSettings(
            base_url=args.base_url,
            endpoint=args.endpoint,
            model=args.judge_model,
            timeout=args.timeout,
            step_limit=args.step_limit,
        ),
        dry_run=args.dry_run,
    )

    client = SolveClient(base_url=args.base_url, endpoint=args.endpoint, timeout=args.timeout)
    settings = SolveSettings(
        base_url=args.base_url,
        endpoint=args.endpoint,
        benchmark="onemillion",
        model=args.model,
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
            system_prompt, user_prompt = build_prompts(entry, include_rubrics=False)
            if args.mode == "memrl":
                user_prompt = _apply_memory(user_prompt, memory_context.get(task_id))
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
