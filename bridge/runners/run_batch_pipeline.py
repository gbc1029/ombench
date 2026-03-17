from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from bridge.dataset.onemillion_loader import load_entries
from bridge.runners.batch_pipeline import BatchPipeline
from experiment.client.solver import SolveClient
from experiment.config.settings import SolveSettings, resolve_model_alias
from experiment.prompt.onemillion_prompt import load_memory_context
from memrl.providers.embedding import OpenAIEmbedder
from memrl.providers.llm import OpenAILLM
from ombench_eval.judge import JudgeSettings, OpenAIJudge

DEFAULT_TRAIN_BASE_URL = "https://holos.openapi-qb.sii.edu.cn"
DEFAULT_TRAIN_ENDPOINT = "/v1/chat/completions"

_log = logging.getLogger(__name__)


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


def _parse_pipeline(value: str) -> set[str]:
    stages = {part.strip() for part in value.split(",") if part.strip()}
    allowed = {"gen", "score", "train"}
    invalid = stages - allowed
    if invalid:
        raise SystemExit(f"Unsupported pipeline stage(s): {', '.join(sorted(invalid))}")
    return stages


def _parse_resume_train(value: str | None) -> str | None:
    if value is None:
        return None
    low = value.strip().lower()
    if low in ("true", "1", "yes"):
        return "true"
    if low in ("false", "0", "no"):
        return "false"
    raise SystemExit(
        f"Invalid --resume-train value: {value!r}. "
        "Use 'true' (resume from trained-output), 'false' (retrain from scratch), "
        "or omit to use the default behaviour."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch pipeline: generate / score / train",
    )
    parser.add_argument("--mode", choices=["plain", "memrl"], default="plain")
    parser.add_argument(
        "--pipeline",
        default="gen,score,train",
        help="Comma-separated stages: gen,score,train (default: gen,score,train)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "datasets" / "OneMillion-Bench",
    )
    parser.add_argument("--base-url", default="http://10.245.198.39:8000")
    parser.add_argument("--endpoint", default="/task/solve")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--memory-context", type=Path, default=None)
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
    parser.add_argument(
        "--judge-retries",
        type=int,
        default=2,
        help="Max attempts for each judge scoring call (default: 2)",
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
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of tasks to process (0 = all)",
    )
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--generated-output",
        type=Path,
        default=Path("outputs/generated.jsonl"),
        help="Output JSONL file for generated responses",
    )
    parser.add_argument(
        "--scored-output",
        type=Path,
        default=Path("outputs/scored.jsonl"),
        help="Output JSONL file for scored responses",
    )
    parser.add_argument(
        "--trained-output",
        type=Path,
        default=Path("outputs/trained.jsonl"),
        help="Output JSONL file for trained task records",
    )
    parser.add_argument(
        "--resume-gen-score",
        type=Path,
        default=None,
        help="Skip already-generated/scored task_ids from this JSONL",
    )
    parser.add_argument(
        "--resume-train",
        default=None,
        help=(
            "Control train-stage resume behaviour. "
            "Omit: auto (train-only resumes from trained-output; with score stage retrains from scratch). "
            "'true': always resume, skip task_ids already in trained-output. "
            "'false': always retrain from scratch (clear trained-output)."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/batch_pipeline"),
        help="Directory to save memory checkpoints",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save checkpoint every N trained items (default: 100)",
    )
    parser.add_argument(
        "--gen-score-batch",
        type=int,
        default=100,
        help=(
            "Batch size for gen+score+train pipeline (default: 100). "
            "Set to 0 to disable batching (process all tasks in one pass)."
        ),
    )
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        default=None,
        help="Load memory checkpoint before memrl generation",
    )
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
                    "backend": "universal_api",
                    "config": {
                        "model_name_or_path": "qwen3-embedding-8b",
                        "provider": "openai",
                        "api_key": api_key,
                        "base_url": "https://og5o9mjcdgckcmejjobdgmjh9ddcjcmk.openapi-qb.sii.edu.cn/v1",
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
        retrieve=RetrieveStrategy.AVEFACT,
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


def _write_jsonl(path: Path, items: list[dict[str, object]], *, append: bool = False) -> None:
    if not items:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(item.get("task_id"))
        for item in _load_jsonl(path)
        if item.get("task_id")
    }


def _clear_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.open("w", encoding="utf-8").close()


def _migrate_file(path: Path, batch_idx: int) -> None:
    """Rename an existing output file to *_batch{N}.ext, then clear the original."""
    if not path.exists() or path.stat().st_size == 0:
        return
    suffix = path.suffix
    stem = path.stem
    dest = path.with_name(f"{stem}_batch{batch_idx}{suffix}")
    path.rename(dest)
    _log.info("Migrated %s -> %s", path, dest)
    _clear_file(path)


def _filter_results(results: list[dict[str, object]], skip_ids: set[str]) -> list[dict[str, object]]:
    if not skip_ids:
        return results
    return [item for item in results if str(item.get("task_id")) not in skip_ids]


def _iter_batches(
    tasks: list[tuple[str, dict[str, object]]],
    batch_size: int,
) -> list[list[tuple[str, dict[str, object]]]]:
    return [tasks[i : i + batch_size] for i in range(0, len(tasks), batch_size)]


def _resolve_latest_snapshot(checkpoint_dir: Path, load_checkpoint: Path | None) -> Path | None:
    if load_checkpoint:
        return load_checkpoint
    snapshot_root = checkpoint_dir / "snapshot"
    if not snapshot_root.exists():
        return None
    candidates = [p for p in snapshot_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def _load_results(path: Path, name: str) -> list[dict[str, object]]:
    if not path.exists():
        raise SystemExit(f"{name} not found: {path}")
    return _load_jsonl(path)


# ---------------------------------------------------------------------------
# Resume-train logic
# ---------------------------------------------------------------------------

def _resolve_train_resume(
    resume_train: str | None,
    stages: set[str],
    trained_output: Path,
) -> tuple[bool, set[str]]:
    """Determine whether to clear trained_output and which ids to skip.

    Returns (should_clear_trained, skip_train_ids).
    """
    has_score = "score" in stages
    has_train = "train" in stages

    if not has_train:
        if resume_train is not None:
            _log.warning(
                "--resume-train is ignored because pipeline does not include 'train'",
            )
        return False, set()

    if resume_train == "false":
        if not has_score:
            _log.warning(
                "--resume-train=false without a score stage: "
                "training from existing scored-output without dedup — "
                "if the scored data overlaps with previous training, "
                "those samples will receive extra weight (trained twice)."
            )
        return True, set()

    if resume_train == "true":
        if has_score:
            _log.warning(
                "--resume-train=true with a score stage: "
                "tasks already in trained-output will be skipped even if "
                "they were re-scored in this run. "
                "Newly scored data whose task_id already appears in "
                "trained-output will NOT participate in training."
            )
        skip = _load_completed_ids(trained_output)
        return False, skip

    # resume_train is None — auto mode
    if has_score:
        return True, set()
    else:
        skip = _load_completed_ids(trained_output)
        return False, skip


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    stages = _parse_pipeline(args.pipeline)
    resume_train = _parse_resume_train(args.resume_train)

    log_path = _build_log_path(args.log_dir, args.log_file)
    handlers = [logging.FileHandler(log_path), logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    _log.info("Logging to %s", log_path)

    # --- Validate pipeline combinations ---
    if "gen" in stages and "train" in stages and "score" not in stages:
        raise SystemExit(
            "Pipeline 'gen,train' (without score) is not allowed. "
            "Training requires scored data to determine success/failure. "
            "Use 'gen,score,train' or run scoring separately first."
        )

    need_entries = "gen" in stages or "score" in stages
    entries = load_entries(args.dataset_dir) if need_entries else {}
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
            max_retries=args.judge_retries,
        ),
        dry_run=args.dry_run,
    )

    need_memory_service = args.mode == "memrl" or "train" in stages
    need_train = "train" in stages
    memory_service: Any = None
    temp_dir_ctx = None

    if need_memory_service:
        train_api_key = os.environ.get(args.train_api_key_env, "")
        if not train_api_key:
            _log.warning(
                "Training API key missing (%s); memory features disabled",
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
            embedder = OpenAIEmbedder(
                api_key=train_api_key,
                base_url="https://og5o9mjcdgckcmejjobdgmjh9ddcjcmk.openapi-qb.sii.edu.cn/v1",
                model="qwen3-embedding-8b",
            )
            try:
                memory_service = _build_memory_service(
                    args, train_llm, embedder, temp_dir, train_api_base=train_api_base,
                )
            except Exception as exc:
                _log.warning(
                    "Failed to init MemoryService (%s); memory features disabled",
                    exc,
                )
                need_train = False

    # --- Resolve resume-train ---
    should_clear_trained, skip_train_ids = _resolve_train_resume(
        resume_train, stages, args.trained_output,
    )

    resume_gen_score = args.resume_gen_score
    skip_gen_score_ids = _load_completed_ids(resume_gen_score) if resume_gen_score else set()

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
            score_workers=args.score_workers,
            generated_output=args.generated_output if "gen" in stages else None,
            scored_output=args.scored_output if "score" in stages else None,
            trained_output=args.trained_output if need_train else None,
        )

        if args.mode == "memrl" and memory_service is not None:
            snapshot_dir = _resolve_latest_snapshot(args.checkpoint_dir, args.load_checkpoint)
            if snapshot_dir:
                try:
                    memory_service.load_checkpoint_snapshot(str(snapshot_dir))
                    _log.info(
                        "Loaded checkpoint snapshot from %s", snapshot_dir,
                    )
                except Exception as exc:
                    _log.warning(
                        "Failed to load checkpoint snapshot (%s)", exc,
                    )

        # ---------------------------------------------------------------
        # Streaming path: gen+score+train interleaved
        # ---------------------------------------------------------------
        if {
            "gen",
            "score",
            "train",
        }.issubset(stages) and need_train:
            tasks = pipeline._build_tasks(entries, resume_gen_score)

            # Clear trained-output upfront when needed
            if should_clear_trained:
                _clear_file(args.trained_output)
            pipeline.init_output_files(append=False)

            if args.gen_score_batch > 0 and len(tasks) > args.gen_score_batch:
                all_results: list[dict[str, object]] = []
                all_trained: list[dict[str, object]] = []
                batches = _iter_batches(tasks, args.gen_score_batch)
                total_batches = len(batches)
                global_total = len(tasks)
                global_offset = 0

                for idx, batch in enumerate(batches):
                    batch_idx = idx + 1

                    # Determine generation mode for this batch
                    if args.mode == "plain":
                        gen_mode = "plain" if idx == 0 else "memrl"
                    else:
                        gen_mode = args.mode

                    if args.mode == "plain" and idx == 0:
                        _log.info(
                            "Batch %d/%d: plain mode (no memory augmentation)",
                            batch_idx, total_batches,
                        )
                    elif args.mode == "plain":
                        _log.info(
                            "Batch %d/%d: memrl mode (using accumulated memories)",
                            batch_idx, total_batches,
                        )

                    # After the first batch: migrate output files to *_batch{N}.ext
                    # then clear originals; resume flags only apply to batch 1
                    if idx > 0:
                        _migrate_file(args.generated_output, idx)
                        _migrate_file(args.scored_output, idx)
                        _migrate_file(args.trained_output, idx)
                        skip_train_ids = set()

                    batch_results, trained_records, last_checkpoint = (
                        pipeline.generate_score_train_tasks(
                            batch,
                            mode=gen_mode,
                            skip_train_ids=skip_train_ids,
                            checkpoint_dir=args.checkpoint_dir,
                            checkpoint_every=args.checkpoint_every,
                            save_final=True,
                            batch_idx=batch_idx,
                            total_batches=total_batches,
                            global_offset=global_offset,
                            global_total=global_total,
                        )
                    )
                    all_results.extend(batch_results)
                    all_trained.extend(trained_records)
                    global_offset += len(batch)

                    if trained_records:
                        skip_train_ids.update(
                            {str(item.get("task_id")) for item in trained_records if item.get("task_id")}
                        )
                    if last_checkpoint and memory_service is not None:
                        snapshot_root = Path(last_checkpoint.get("cube_dir", "")).parent
                        if snapshot_root.exists():
                            try:
                                memory_service.load_checkpoint_snapshot(str(snapshot_root))
                            except Exception as exc:
                                _log.warning(
                                    "Failed to reload checkpoint (%s)", exc,
                                )

                pipeline.print_summary(all_results)
            else:
                results, trained_records, last_checkpoint = (
                    pipeline.generate_score_train_tasks(
                        tasks,
                        mode=args.mode,
                        skip_train_ids=skip_train_ids,
                        checkpoint_dir=args.checkpoint_dir,
                        checkpoint_every=args.checkpoint_every,
                        save_final=True,
                    )
                )
                pipeline.print_summary(results)
            return

        # ---------------------------------------------------------------
        # Non-batched paths
        # ---------------------------------------------------------------

        # Clear trained-output upfront when needed
        if should_clear_trained and need_train:
            _clear_file(args.trained_output)

        pipeline.init_output_files(append=False)

        if "gen" in stages and "score" in stages:
            results = pipeline.generate_and_score(entries, mode=args.mode, resume_from=resume_gen_score)
            pipeline.print_summary(results)
            if need_train:
                filtered = _filter_results(results, skip_train_ids)
                pipeline.train(
                    filtered,
                    skip_ids=skip_train_ids,
                    checkpoint_dir=args.checkpoint_dir,
                    checkpoint_every=args.checkpoint_every,
                    save_final=True,
                )
            return

        if "gen" in stages and "score" not in stages:
            results = pipeline.generate(entries, mode=args.mode, resume_from=resume_gen_score)
            return

        if "score" in stages and "gen" not in stages:
            if not entries:
                raise SystemExit("Scoring requires dataset entries; enable gen/score stages")
            results = _load_results(args.generated_output, "Generated output")
            results = _filter_results(results, skip_gen_score_ids)
            scored = pipeline.score(results, entries)
            _write_jsonl(args.scored_output, scored)
            pipeline.print_summary(scored)
            if need_train:
                filtered = _filter_results(scored, skip_train_ids)
                pipeline.train(
                    filtered,
                    skip_ids=skip_train_ids,
                    checkpoint_dir=args.checkpoint_dir,
                    checkpoint_every=args.checkpoint_every,
                    save_final=True,
                )
            return

        if "train" in stages:
            scored = _load_results(args.scored_output, "Scored output")
            filtered = _filter_results(scored, skip_train_ids)
            if need_train:
                pipeline.train(
                    filtered,
                    skip_ids=skip_train_ids,
                    checkpoint_dir=args.checkpoint_dir,
                    checkpoint_every=args.checkpoint_every,
                    save_final=True,
                )
            return

    finally:
        if temp_dir_ctx is not None:
            temp_dir_ctx.cleanup()


if __name__ == "__main__":
    main()
