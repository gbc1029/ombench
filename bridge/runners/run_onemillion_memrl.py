from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, Optional

from experiment.client.solver import SolveClient
from experiment.config.settings import SolveSettings
from experiment.prompt.onemillion_prompt import build_prompts
from bridge.adapters.solve_llm import SolveLLM
from bridge.adapters.hash_embedder import HashEmbedder
from bridge.dataset.onemillion_loader import load_entries
from memrl.service.memory_service import MemoryService
from memrl.service.strategies import BuildStrategy, RetrieveStrategy, UpdateStrategy, StrategyConfiguration


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
    parser.add_argument("--model", default="sii-holos/Qwen 3.5 397B A17B")
    parser.add_argument("--request-id", default="omb-memrl")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--step-limit", type=int, default=150)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_entries(args.dataset_dir)

    client = SolveClient(base_url=args.base_url, endpoint=args.endpoint, timeout=args.timeout)
    settings = SolveSettings(
        base_url=args.base_url,
        endpoint=args.endpoint,
        benchmark="onemillion",
        model=args.model,
        timeout=args.timeout,
        step_limit=args.step_limit,
        request_id=args.request_id,
        system_prompt="",
        user_prompt="",
    )
    llm = SolveLLM(client=client, settings=settings, dry_run=args.dry_run)
    embedder = HashEmbedder()

    with tempfile.TemporaryDirectory(prefix="onemillion_memrl_") as temp_dir:
        mos_config = _write_mos_config(
            Path(temp_dir),
            api_key="placeholder",
            base_url=args.base_url,
            model=args.model,
        )
        strategy = StrategyConfiguration(
            build=BuildStrategy.PROCEDURALIZATION,
            retrieve=RetrieveStrategy.QUERY,
            update=UpdateStrategy.ADJUSTMENT,
        )
        memory = MemoryService(
            mos_config_path=str(mos_config),
            llm_provider=llm,
            embedding_provider=embedder,
            strategy_config=strategy,
            user_id="onemillion_memrl",
            enable_value_driven=True,
            max_keywords=8,
        )

        for idx, (task_id, entry) in enumerate(entries.items(), start=1):
            if args.limit and idx > args.limit:
                break
            system_prompt, user_prompt = build_prompts(entry)
            response = llm.solve_raw(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                task_id=task_id,
            )
            status = response.get("status") if isinstance(response, dict) else None
            trajectory = json.dumps(response.get("session_data", {}).get("messages", []), ensure_ascii=False)
            memory.add_memory(
                task_description=user_prompt,
                trajectory=trajectory,
                success=status == "completed",
                metadata={"task_id": task_id, "request_id": response.get("request_id"), "status": status},
            )


if __name__ == "__main__":
    main()
