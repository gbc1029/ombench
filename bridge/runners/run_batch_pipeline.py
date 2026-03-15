from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path

from bridge.adapters.hash_embedder import HashEmbedder
from bridge.dataset.onemillion_loader import load_entries
from bridge.runners.batch_pipeline import BatchPipeline
from experiment.client.solver import SolveClient
from experiment.config.settings import SolveSettings, resolve_model_alias
from experiment.prompt.onemillion_prompt import load_memory_context
from memrl.providers.llm import OpenAILLM
from ombench_eval.judge import JudgeSettings, OpenAIJudge

DEFAULT_TRAIN_BASE_URL = "https://holos.openapi-qb.sii.edu.cn"
DEFAULT_TRAIN_ENDPOINT = "/v1/chat/completions"


def _resolve_openai_base_url(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    if endpoint.startswith("/v1") or "/v1/" in endpoint:
        return f"{base}/v1"
    return base




def _build_log_path(log_dir: Path, log_file: Path | None) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    if log_file is None:
        filename = f"run_batch_pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
        return log_dir / filename
    log_file = Path(log_file)
    if log_file.is_absolute():
        return log_file
    return log_dir / log_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch pipeline: generate -> score -> train",
    )
    parser.add_argument("--mode", choices=["plain", "memrl", "rubric"], default="plain")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "datasets" / "OneMillion-Bench",
    )
    parser.add_argument("--base-url", default="http://10.245.198.39:8000")
    parser.add_argument("--endpoint", default="/task/solve")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--memory-context", type=Path, default=None)
    parser.add_argument("--responses-file", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument(
        "--train-base-url",
        default=DEFAULT_TRAIN_BASE_URL,
        help="Training LLM base URL (default: judge base URL)",
    )
    parser.add_argument(
        "--train-endpoint",
        default=DEFAULT_TRAIN_ENDPOINT,
        help="Training LLM endpoint (default: /v1/chat/completions)",
    )
    parser.add_argument("--train-model", default="qwen3.5-397b-a17b")
    parser.add_argument("--train-api-key-env", default="INF_API_KEY")
    parser.add_argument(
        "--judge-model",
        default="qwen3.5-397b-a17b",
        help="Model for judging (default: qwen3.5-397b-a17b)",
    )
    parser.add_argument(
        "--judge-base-url",
        default=DEFAULT_TRAIN_BASE_URL,
        help="Base URL for judge API (default: holos openapi)",
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
    parser.add_argument("--step-limit", type=int, default=150)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrency level for generate stage",
    )
    parser.add_argument(
        "--score-workers",
        type=int,
        default=4,
        help="Concurrency level for scoring stage (default: 4)",
    )

    parser.add_argument("--limit", type=int, default=0, help="Max number of tasks to process (0 = all)")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--no-train", action="store_true", help="Skip the MemRL training stage")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/results.jsonl"), help="Output JSONL file for results")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a previous output JSONL (skip completed task_ids)")
    return parser.parse_args()


def _build_memory_service(args, llm, embedder, temp_dir, *, train_api_base: str):
    from memrl.service.memory_service import MemoryService
    from memrl.service.strategies import (
        BuildStrategy, RetrieveStrategy, StrategyConfiguration, UpdateStrategy,
    )

    api_key = os.environ.get(args.train_api_key_env, "placeholder")
    config = {
        "chat_model": {
            "backend": "openai",
            "config": {
                "model_name_or_path": args.train_model,
                "api_key": api_key,
                "api_base": train_api_base,
            },
        },
        "mem_reader": {
            "backend": "simple_struct",
            "config": {
                "llm": {
                    "backend": "openai",
                    "config": {
                        "model_name_or_path": args.train_model,
                        "api_key": api_key,
                        "api_base": train_api_base,
                    },
                },
                "embedder": {
                    "backend": "sentence_transformer",
                    "config": {
                        "model_name_or_path": "all-MiniLM-L6-v2",
                    },
                },
                "chunker": {"backend": "sentence", "config": {"chunk_size": 500}},
            },
        },
        "user_manager": {
            "backend": "sqlite",
            "config": {"db_path": str(Path(temp_dir) / "users.db")},
        },
        "top_k": 5,
    }
    config_path = Path(temp_dir) / "mos_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    strategy = StrategyConfiguration(
        build=BuildStrategy.PROCEDURALIZATION,
        retrieve=RetrieveStrategy.QUERY,
        update=UpdateStrategy.ADJUSTMENT,
    )
    return MemoryService(
        mos_config_path=str(config_path),
        llm_provider=llm,
        embedding_provider=embedder,
        strategy_config=strategy,
        user_id="batch_pipeline",
        enable_value_driven=True,
        max_keywords=8,
    )


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main() -> None:
    args = parse_args()
    log_path = _build_log_path(args.log_dir, args.log_file)
    handlers = [logging.FileHandler(log_path), logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    logging.getLogger(__name__).info("Logging to %s", log_path)

    entries = load_entries(args.dataset_dir)
    memory_context = load_memory_context(args.memory_context) if args.memory_context else {}

    client = SolveClient(
        base_url=args.base_url, endpoint=args.endpoint, timeout=args.timeout,
    )
    settings = SolveSettings(
        base_url=args.base_url,
        endpoint=args.endpoint,
        benchmark="onemillion",
        model=resolve_model_alias(args.model),
        timeout=args.timeout,
        step_limit=args.step_limit,
        request_id="omb-batch",
        system_prompt="",
        user_prompt="",
    )
    judge = OpenAIJudge(
        JudgeSettings(
            base_url=args.judge_base_url,
            model=args.judge_model,
            timeout=args.judge_timeout,
        ),
        dry_run=args.dry_run,
    )

    need_train = (not args.no_train) and args.mode != "rubric"
    memory_service = None
    temp_dir_ctx = None

    if need_train:
        train_api_key = os.environ.get(args.train_api_key_env, "")
        if not train_api_key:
            logging.getLogger(__name__).warning(
                "Training API key missing (%s); train stage will be skipped",
                args.train_api_key_env,
            )
            need_train = False
        else:
            temp_dir_ctx = tempfile.TemporaryDirectory(prefix="batch_pipeline_")
            temp_dir = temp_dir_ctx.name
            train_api_base = _resolve_openai_base_url(
                args.train_base_url, args.train_endpoint,
            )
            train_llm = OpenAILLM(
                api_key=train_api_key,
                base_url=train_api_base,
                model=args.train_model,
                default_temperature=0.1,
                default_max_tokens=4096,
            )
            embedder = HashEmbedder()
            try:
                memory_service = _build_memory_service(
                    args, train_llm, embedder, temp_dir, train_api_base=train_api_base,
                )
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Failed to init MemoryService (%s), train stage will be skipped",
                    exc,
                )
                need_train = False

    try:
        pipeline = BatchPipeline(
            client=client,
            settings=settings,
            judge=judge,
            memory_service=memory_service,
            memory_context=memory_context,
            workers=args.workers,
            limit=args.limit,
            dry_run=args.dry_run,
            max_retries=args.max_retries,
            output=args.output,
            score_workers=args.score_workers,
        )

        if args.mode == "rubric":
            if not args.responses_file:
                raise SystemExit("--responses-file is required for rubric mode")
            responses = _load_jsonl(args.responses_file)
            results = []
            for item in responses:
                task_id = str(item.get("task_id", ""))
                answer = item.get("answer", "")
                if task_id:
                    results.append({"task_id": task_id, "answer": answer})
            results = pipeline.score(results, entries)
            pipeline._save_results(results)
            pipeline.print_summary(results)
        else:
            pipeline.run(
                entries,
                mode=args.mode,
                skip_train=not need_train,
                resume_from=args.resume,
            )
    finally:
        if temp_dir_ctx is not None:
            temp_dir_ctx.cleanup()


if __name__ == "__main__":
    main()
