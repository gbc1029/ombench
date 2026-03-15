from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

from bridge.adapters.hash_embedder import HashEmbedder
from bridge.adapters.solve_llm import SolveLLM
from bridge.dataset.onemillion_loader import load_entries
from bridge.runners.batch_pipeline import BatchPipeline
from experiment.client.solver import SolveClient
from experiment.config.settings import SolveSettings
from ombench_eval.judge import JudgeSettings, SolveJudge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch pipeline: generate -> score -> train",
    )
    parser.add_argument("--mode", choices=["direct", "memrl"], default="direct",
                        help="direct: use experiment/ client; memrl: use SolveLLM adapter")
    parser.add_argument("--dataset-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "datasets" / "OneMillion-Bench")
    parser.add_argument("--base-url", default="http://10.245.198.39:8000")
    parser.add_argument("--endpoint", default="/task/solve")
    parser.add_argument("--model", default="sii-holos/Qwen 3.5 397B A17B")
    parser.add_argument("--judge-model", default=None,
                        help="Model for judging (defaults to --model)")
    parser.add_argument("--timeout", type=int, default=1200,
                        help="Timeout for generation requests (default: 1200)")
    parser.add_argument("--judge-timeout", type=int, default=600,
                        help="Timeout for scoring/judge requests (default: 600)")
    parser.add_argument("--step-limit", type=int, default=150)
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrency level for generate stage")
    parser.add_argument("--score-batch-size", type=int, default=1,
                        help="Number of items per scoring request (default: 1)")
    parser.add_argument("--score-workers", type=int, default=4,
                        help="Concurrency level for scoring stage (default: 4)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max number of tasks to process (0 = all)")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--no-train", action="store_true",
                        help="Skip the MemRL training stage")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/results.jsonl"),
                        help="Output JSONL file for results")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from a previous output JSONL (skip completed task_ids)")
    return parser.parse_args()


def _build_memory_service(args, llm, embedder, temp_dir):
    from memrl.service.memory_service import MemoryService
    from memrl.service.strategies import (
        BuildStrategy, RetrieveStrategy, StrategyConfiguration, UpdateStrategy,
    )

    config = {
        "chat_model": {
            "backend": "openai",
            "config": {
                "model_name_or_path": args.model,
                "api_key": "placeholder",
                "api_base": args.base_url,
            },
        },
        "mem_reader": {
            "backend": "simple_struct",
            "config": {
                "llm": {
                    "backend": "openai",
                    "config": {
                        "model_name_or_path": args.model,
                        "api_key": "placeholder",
                        "api_base": args.base_url,
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    entries = load_entries(args.dataset_dir)
    judge_model = args.judge_model or args.model

    client = SolveClient(
        base_url=args.base_url, endpoint=args.endpoint, timeout=args.timeout,
    )
    settings = SolveSettings(
        base_url=args.base_url, endpoint=args.endpoint, benchmark="onemillion",
        model=args.model, timeout=args.timeout, step_limit=args.step_limit,
        request_id="omb-batch", system_prompt="", user_prompt="",
    )
    judge = SolveJudge(
        JudgeSettings(
            base_url=args.base_url, endpoint=args.endpoint,
            model=judge_model, timeout=args.judge_timeout, step_limit=args.step_limit,
        ),
        dry_run=args.dry_run,
    )

    llm = SolveLLM(client=client, settings=settings, dry_run=args.dry_run)
    embedder = HashEmbedder()

    need_train = not args.no_train
    memory_service = None
    temp_dir_ctx = None

    if need_train:
        temp_dir_ctx = tempfile.TemporaryDirectory(prefix="batch_pipeline_")
        temp_dir = temp_dir_ctx.name
        try:
            memory_service = _build_memory_service(args, llm, embedder, temp_dir)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Failed to init MemoryService (%s), train stage will be skipped", exc,
            )
            need_train = False

    try:
        pipeline = BatchPipeline(
            client=client, settings=settings, judge=judge, llm=llm,
            memory_service=memory_service, workers=args.workers,
            limit=args.limit, dry_run=args.dry_run,
            max_retries=args.max_retries, output=args.output,
            score_batch_size=args.score_batch_size,
            score_workers=args.score_workers,
        )
        pipeline.run(
            entries, mode=args.mode, skip_train=not need_train,
            resume_from=args.resume,
        )
    finally:
        if temp_dir_ctx is not None:
            temp_dir_ctx.cleanup()


if __name__ == "__main__":
    main()
