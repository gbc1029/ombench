from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path

from bridge.adapters.hash_embedder import HashEmbedder
from bridge.adapters.solve_llm import SolveLLM
from bridge.dataset.onemillion_loader import load_entries
from experiment.client.solver import SolveClient
from experiment.config.settings import SolveSettings, resolve_model_alias
from experiment.prompt.onemillion_prompt import build_plain_prompts
from memrl.providers.llm import OpenAILLM
from memrl.service.memory_service import MemoryService
from memrl.service.strategies import (
    BuildStrategy,
    RetrieveStrategy,
    StrategyConfiguration,
    UpdateStrategy,
)

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
        filename = f"run_onemillion_memrl_{time.strftime('%Y%m%d_%H%M%S')}.log"
        return log_dir / filename
    log_file = Path(log_file)
    if log_file.is_absolute():
        return log_file
    return log_dir / log_file


def _write_mos_config(temp_dir: Path, *, api_key: str, base_url: str, model: str) -> Path:
    config = {
        "chat_model": {
            "backend": "openai",
            "config": {
                "model_name_or_path": model,
                "api_key": api_key,
                "api_base": base_url,
            },
        },
        "mem_reader": {
            "backend": "simple_struct",
            "config": {
                "llm": {
                    "backend": "openai",
                    "config": {
                        "model_name_or_path": model,
                        "api_key": api_key,
                        "api_base": base_url,
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
            "config": {"db_path": str(temp_dir / "users.db")},
        },
        "top_k": 5,
    }
    config_path = temp_dir / "mos_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MemRL on OneMillion-Bench tasks")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "datasets" / "OneMillion-Bench",
        help="Path to onemillion dataset directory",
    )
    parser.add_argument("--base-url", default="http://10.245.198.39:8000")
    parser.add_argument("--endpoint", default="/task/solve")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--request-id", default="omb-memrl")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--step-limit", type=int, default=150)
    parser.add_argument("--limit", type=int, default=0)
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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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
    client = SolveClient(base_url=args.base_url, endpoint=args.endpoint, timeout=args.timeout)
    settings = SolveSettings(
        base_url=args.base_url,
        endpoint=args.endpoint,
        benchmark="onemillion",
        model=resolve_model_alias(args.model),
        timeout=args.timeout,
        step_limit=args.step_limit,
        request_id=args.request_id,
        system_prompt="",
        user_prompt="",
    )
    llm = SolveLLM(client=client, settings=settings, dry_run=args.dry_run)

    train_api_key = os.environ.get(args.train_api_key_env, "")
    if not train_api_key:
        raise SystemExit(f"Training API key missing: {args.train_api_key_env}")

    train_api_base = _resolve_openai_base_url(args.train_base_url, args.train_endpoint)
    train_llm = OpenAILLM(
        api_key=train_api_key,
        base_url=train_api_base,
        model=args.train_model,
        default_temperature=0.1,
        default_max_tokens=4096,
    )
    embedder = HashEmbedder()

    with tempfile.TemporaryDirectory(prefix="onemillion_memrl_") as temp_dir:
        mos_config = _write_mos_config(
            Path(temp_dir),
            api_key=train_api_key,
            base_url=train_api_base,
            model=args.train_model,
        )
        strategy = StrategyConfiguration(
            build=BuildStrategy.PROCEDURALIZATION,
            retrieve=RetrieveStrategy.QUERY,
            update=UpdateStrategy.ADJUSTMENT,
        )
        memory = MemoryService(
            mos_config_path=str(mos_config),
            llm_provider=train_llm,
            embedding_provider=embedder,
            strategy_config=strategy,
            user_id="onemillion_memrl",
            enable_value_driven=True,
            max_keywords=8,
        )

        for idx, (task_id, entry) in enumerate(entries.items(), start=1):
            if args.limit and idx > args.limit:
                break
            system_prompt, user_prompt = build_plain_prompts(entry)
            response = llm.solve_raw(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                task_id=task_id,
            )
            status = response.get("status") if isinstance(response, dict) else None
            trajectory = json.dumps(
                response.get("session_data", {}).get("messages", []),
                ensure_ascii=False,
            )
            memory.add_memory(
                task_description=user_prompt,
                trajectory=trajectory,
                success=status == "completed",
                metadata={
                    "task_id": task_id,
                    "request_id": response.get("request_id"),
                    "task_status": status,
                },
            )


if __name__ == "__main__":
    main()
