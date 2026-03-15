from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

from bridge.adapters.solve_llm import SolveLLM
from bridge.dataset.onemillion_loader import load_entries
from experiment.client.solver import SolveClient
from experiment.config.settings import SolveSettings
from experiment.prompt.onemillion_prompt import build_prompts
from ombench_eval.evaluator import score_response, score_responses_batch
from ombench_eval.judge import BaseJudge

logger = logging.getLogger(__name__)


def _extract_answer(response: Dict[str, Any]) -> str:
    if not isinstance(response, dict):
        return ""
    result = response.get("result")
    if isinstance(result, dict):
        answer = result.get("answer")
        if isinstance(answer, str):
            return answer
    return ""


def _load_completed_ids(path: Path) -> set:
    ids: set = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["task_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


class BatchPipeline:
    def __init__(
        self,
        *,
        client: SolveClient,
        settings: SolveSettings,
        judge: BaseJudge,
        llm: Optional[SolveLLM] = None,
        memory_service: Optional[Any] = None,
        workers: int = 4,
        limit: int = 0,
        dry_run: bool = False,
        max_retries: int = 1,
        output: Optional[Path] = None,
        score_batch_size: int = 1,
        score_workers: int = 4,
    ) -> None:
        self.client = client
        self.settings = settings
        self.judge = judge
        self.llm = llm
        self.memory_service = memory_service
        self.workers = max(workers, 1)
        self.limit = limit
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.output = output
        self.score_batch_size = max(score_batch_size, 1)
        self.score_workers = max(score_workers, 1)

    def _generate_one_direct(self, task_id: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt, user_prompt = build_prompts(entry, include_rubrics=False)
        payload = self.settings.build_payload(
            overrides={
                "task_id": task_id,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            },
        )
        response = self.client.solve_with_retry(
            payload, dry_run=self.dry_run, max_retries=self.max_retries,
        )
        return {
            "task_id": task_id,
            "response": response,
            "answer": _extract_answer(response),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

    def _generate_one_memrl(self, task_id: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        if self.llm is None:
            raise RuntimeError("SolveLLM is required for memrl mode")
        system_prompt, user_prompt = build_prompts(entry, include_rubrics=False)
        response = self.llm.solve_raw(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            task_id=task_id,
        )
        return {
            "task_id": task_id,
            "response": response,
            "answer": _extract_answer(response),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

    def generate(
        self,
        entries: Dict[str, Dict[str, Any]],
        *,
        mode: str = "direct",
        resume_from: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        worker_fn = self._generate_one_direct if mode == "direct" else self._generate_one_memrl

        skip_ids = _load_completed_ids(resume_from) if resume_from else set()
        if skip_ids:
            logger.info("Resuming: skipping %d already-completed task(s)", len(skip_ids))

        tasks = [(task_id, entry) for task_id, entry in entries.items() if task_id not in skip_ids]
        if tasks:
            random.shuffle(tasks)
        if self.limit:
            tasks = tasks[:self.limit]

        results: List[Dict[str, Any]] = []
        logger.info("Generating responses: %d tasks, %d workers, mode=%s", len(tasks), self.workers, mode)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            future_map = {pool.submit(worker_fn, tid, ent): tid for tid, ent in tasks}
            with tqdm(total=len(future_map), desc="Generate", unit="task") as pbar:
                for future in as_completed(future_map):
                    tid = future_map[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as exc:
                        logger.error("Task %s generation failed: %s", tid, exc)
                        results.append({
                            "task_id": tid, "response": None, "answer": "",
                            "error": str(exc), "stage": "generate",
                        })
                    pbar.update(1)

        return results

    def _score_one(self, item: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
        answer = item.get("answer", "")
        if not answer or item.get("error"):
            item["score"] = None
            return item
        score = score_response(
            judge=self.judge,
            question=entry.get("question", ""),
            response=answer,
            rubrics=entry.get("rubrics", []),
            system_prompt=entry.get("system_prompt"),
        )
        item["score"] = score
        return item

    def score(
        self,
        results: List[Dict[str, Any]],
        entries: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        scoreable_indices: List[int] = []
        score_items: List[Dict[str, Any]] = []
        for i, r in enumerate(results):
            if r.get("answer") and not r.get("error"):
                entry = entries.get(r["task_id"], {})
                score_items.append({
                    "question": entry.get("question", ""),
                    "response": r.get("answer", ""),
                    "rubrics": entry.get("rubrics", []),
                    "system_prompt": entry.get("system_prompt"),
                })
                scoreable_indices.append(i)

        logger.info(
            "Scoring responses: %d / %d tasks, batch_size=%d, workers=%d",
            len(score_items), len(results), self.score_batch_size, self.score_workers,
        )

        if not score_items:
            return results

        batch_scores = score_responses_batch(
            judge=self.judge,
            items=score_items,
            batch_size=self.score_batch_size,
            max_workers=self.score_workers,
        )

        for pos, idx in enumerate(scoreable_indices):
            if pos < len(batch_scores):
                results[idx]["score"] = batch_scores[pos]
            else:
                results[idx]["score"] = None

        return results

    def train(self, results: List[Dict[str, Any]]) -> int:
        if self.memory_service is None:
            logger.warning("No memory_service provided, skipping train stage")
            return 0

        trained = 0
        for item in tqdm(results, desc="Train", unit="task"):
            if item.get("error") or not item.get("answer"):
                continue
            response = item.get("response", {})
            status = response.get("status") if isinstance(response, dict) else None
            trajectory = json.dumps(
                response.get("session_data", {}).get("messages", [])
                if isinstance(response, dict) else [],
                ensure_ascii=False,
            )
            score_info = item.get("score", {})
            is_success = status == "completed"
            if isinstance(score_info, dict) and score_info.get("max_score"):
                ratio = score_info.get("score", 0) / score_info["max_score"]
                is_success = is_success or ratio >= 0.5

            try:
                self.memory_service.add_memory(
                    task_description=item.get("user_prompt", ""),
                    trajectory=trajectory,
                    success=is_success,
                    metadata={
                        "task_id": item["task_id"],
                        "request_id": response.get("request_id") if isinstance(response, dict) else None,
                        "status": status,
                        "score": score_info,
                    },
                )
                trained += 1
            except Exception as exc:
                logger.error("Task %s train failed: %s", item["task_id"], exc)

        logger.info("Trained %d / %d tasks", trained, len(results))
        return trained

    def _save_results(self, results: List[Dict[str, Any]]) -> None:
        if not self.output:
            return
        with self.output.open("w", encoding="utf-8") as f:
            for item in results:
                safe = {
                    "task_id": item.get("task_id"),
                    "answer": item.get("answer"),
                    "score": item.get("score"),
                    "error": item.get("error"),
                    "stage": item.get("stage"),
                }
                if not self.dry_run:
                    safe["response"] = item.get("response")
                f.write(json.dumps(safe, ensure_ascii=False) + "\n")
        logger.info("Results saved to %s", self.output)

    def print_summary(self, results: List[Dict[str, Any]]) -> None:
        total = len(results)
        errors = [r for r in results if r.get("error")]
        scored = [r for r in results if isinstance(r.get("score"), dict) and r["score"].get("max_score")]

        print(f"\n{'=' * 40}")
        print(f"  Total: {total} | Completed: {total - len(errors)} | Failed: {len(errors)}")

        if scored:
            avg_score = sum(r["score"]["score"] for r in scored) / len(scored)
            avg_max = sum(r["score"]["max_score"] for r in scored) / len(scored)
            pct = (avg_score / avg_max * 100) if avg_max > 0 else 0
            print(f"  Avg Score: {avg_score:.1f}/{avg_max:.1f} ({pct:.1f}%)")

        by_subset: Dict[str, List[Dict]] = defaultdict(list)
        for r in results:
            parts = r.get("task_id", "").split("/")
            subset = parts[0] if parts else "unknown"
            by_subset[subset].append(r)

        if len(by_subset) > 1:
            print("  By subset:")
            for subset, items in sorted(by_subset.items()):
                ok = sum(1 for i in items if not i.get("error"))
                sub_scored = [i for i in items if isinstance(i.get("score"), dict) and i["score"].get("max_score")]
                if sub_scored:
                    s = sum(i["score"]["score"] for i in sub_scored) / len(sub_scored)
                    m = sum(i["score"]["max_score"] for i in sub_scored) / len(sub_scored)
                    print(f"    {subset}: {ok}/{len(items)} completed, avg {s:.1f}/{m:.1f}")
                else:
                    print(f"    {subset}: {ok}/{len(items)} completed, no scores")
        print(f"{'=' * 40}\n")

    def run(
        self,
        entries: Dict[str, Dict[str, Any]],
        *,
        mode: str = "direct",
        skip_train: bool = False,
        resume_from: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        results = self.generate(entries, mode=mode, resume_from=resume_from)
        results = self.score(results, entries)
        if not skip_train:
            self.train(results)
        self._save_results(results)
        self.print_summary(results)
        return results
