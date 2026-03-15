from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional


from tqdm import tqdm

from experiment.client.solver import SolveClient
from experiment.config.settings import SolveSettings
from experiment.prompt.onemillion_prompt import apply_memory, build_plain_prompts
from ombench_eval.evaluator import score_response, score_responses_batch
from ombench_eval.judge import BaseJudge

logger = logging.getLogger(__name__)


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    message = str(exc).lower()
    return "timeout" in message or "timed out" in message


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
        memory_service: Optional[Any] = None,
        memory_context: Optional[Dict[str, str]] = None,
        workers: int = 4,
        limit: int = 0,
        dry_run: bool = False,
        max_retries: int = 1,
        output: Optional[Path] = None,
        score_workers: int = 4,
    ) -> None:
        self.client = client
        self.settings = settings
        self.judge = judge
        self.memory_service = memory_service
        self.memory_context = memory_context or {}
        self.workers = max(workers, 1)
        self.limit = limit
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.output = output
        self.score_workers = max(score_workers, 1)

    def _score_item(self, item: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
        try:
            score = score_response(
                judge=self.judge,
                question=entry.get("question", ""),
                response=item.get("answer", ""),
                rubrics=entry.get("rubrics", []),
                system_prompt=entry.get("system_prompt"),
            )
            item["score"] = score
        except Exception as exc:
            if _is_timeout_error(exc):
                logger.warning("Score timeout for task_id: %s", item.get("task_id"))
            logger.error("Task %s score failed: %s", item.get("task_id"), exc)
            item["score"] = {"raw": str(exc)}
        return item

    def _build_prompts(
        self,
        task_id: str,
        entry: Dict[str, Any],
        *,
        mode: str,
    ) -> Dict[str, str]:
        system_prompt, user_prompt = build_plain_prompts(entry)
        if mode == "memrl":
            user_prompt = apply_memory(user_prompt, self.memory_context.get(task_id))
        return {"system_prompt": system_prompt, "user_prompt": user_prompt}

    def _generate_one_direct(
        self,
        task_id: str,
        entry: Dict[str, Any],
        *,
        mode: str,
    ) -> Dict[str, Any]:
        prompts = self._build_prompts(task_id, entry, mode=mode)
        payload = self.settings.build_payload(
            overrides={
                "task_id": task_id,
                "system_prompt": prompts["system_prompt"],
                "user_prompt": prompts["user_prompt"],
            },
        )
        response = self.client.solve_with_retry(
            payload, dry_run=self.dry_run, max_retries=self.max_retries,
        )
        return {
            "task_id": task_id,
            "response": response,
            "answer": _extract_answer(response),
            "system_prompt": prompts["system_prompt"],
            "user_prompt": prompts["user_prompt"],
        }

    def generate(
        self,
        entries: Dict[str, Dict[str, Any]],
        *,
        mode: str = "plain",
        resume_from: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        skip_ids = _load_completed_ids(resume_from) if resume_from else set()
        if skip_ids:
            logger.info("Resuming: skipping %d already-completed task(s)", len(skip_ids))

        tasks = [
            (task_id, entry)
            for task_id, entry in entries.items()
            if task_id not in skip_ids
        ]
        if tasks:
            random.shuffle(tasks)
        if self.limit:
            tasks = tasks[: self.limit]

        results: List[Dict[str, Any]] = []
        logger.info(
            "Generating responses: %d tasks, %d workers, mode=%s",
            len(tasks),
            self.workers,
            mode,
        )

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            future_map = {
                pool.submit(self._generate_one_direct, tid, ent, mode=mode): tid
                for tid, ent in tasks
            }
            with tqdm(total=len(future_map), desc="Generate", unit="task") as pbar:
                for future in as_completed(future_map):
                    tid = future_map[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as exc:
                        logger.error("Task %s generation failed: %s", tid, exc)
                        results.append(
                            {
                                "task_id": tid,
                                "response": None,
                                "answer": "",
                                "error": str(exc),
                                "stage": "generate",
                            }
                        )
                    pbar.update(1)

        return results

    def generate_and_score(
        self,
        entries: Dict[str, Dict[str, Any]],
        *,
        mode: str = "plain",
        resume_from: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        skip_ids = _load_completed_ids(resume_from) if resume_from else set()
        if skip_ids:
            logger.info("Resuming: skipping %d already-completed task(s)", len(skip_ids))

        tasks = [
            (task_id, entry)
            for task_id, entry in entries.items()
            if task_id not in skip_ids
        ]
        if tasks:
            random.shuffle(tasks)
        if self.limit:
            tasks = tasks[: self.limit]

        results: List[Dict[str, Any]] = []
        logger.info(
            "Generating responses: %d tasks, %d workers, mode=%s",
            len(tasks),
            self.workers,
            mode,
        )
        logger.info(
            "Scoring responses: %d tasks, workers=%d",
            len(tasks),
            self.score_workers,
        )

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            future_map = {
                pool.submit(self._generate_one_direct, tid, ent, mode=mode): tid
                for tid, ent in tasks
            }
            with (
                tqdm(total=len(future_map), desc="Generate", unit="task") as gen_bar,
                tqdm(total=len(future_map), desc="Score", unit="task") as score_bar,
            ):
                for future in as_completed(future_map):
                    tid = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        logger.error("Task %s generation failed: %s", tid, exc)
                        result = {
                            "task_id": tid,
                            "response": None,
                            "answer": "",
                            "error": str(exc),
                            "stage": "generate",
                            "score": None,
                        }
                    else:
                        if result.get("answer") and not result.get("error"):
                            entry = entries.get(tid, {})
                            result = self._score_item(result, entry)
                        else:
                            result["score"] = None
                    results.append(result)
                    gen_bar.update(1)
                    score_bar.update(1)

        return results

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
                score_items.append(
                    {
                        "task_id": r.get("task_id"),
                        "question": entry.get("question", ""),
                        "response": r.get("answer", ""),
                        "rubrics": entry.get("rubrics", []),
                        "system_prompt": entry.get("system_prompt"),
                    }
                )
                scoreable_indices.append(i)

        logger.info(
            "Scoring responses: %d / %d tasks, workers=%d",
            len(score_items),
            len(results),
            self.score_workers,
        )


        if not score_items:
            return results

        batch_scores = score_responses_batch(
            judge=self.judge,
            items=score_items,
            batch_size=1,
            max_workers=self.score_workers,
            show_progress=True,
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
                if isinstance(response, dict)
                else [],
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
                        "request_id": response.get("request_id")
                        if isinstance(response, dict)
                        else None,
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
        scored = [
            r
            for r in results
            if isinstance(r.get("score"), dict) and r["score"].get("max_score")
        ]

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
                sub_scored = [
                    i
                    for i in items
                    if isinstance(i.get("score"), dict) and i["score"].get("max_score")
                ]
                if sub_scored:
                    s = sum(i["score"]["score"] for i in sub_scored) / len(sub_scored)
                    m = sum(i["score"]["max_score"] for i in sub_scored) / len(sub_scored)
                    print(
                        f"    {subset}: {ok}/{len(items)} completed, avg {s:.1f}/{m:.1f}"
                    )
                else:
                    print(f"    {subset}: {ok}/{len(items)} completed, no scores")
        print(f"{'=' * 40}\n")

    def run(
        self,
        entries: Dict[str, Dict[str, Any]],
        *,
        mode: str = "plain",
        skip_train: bool = False,
        resume_from: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        results = self.generate_and_score(entries, mode=mode, resume_from=resume_from)
        if not skip_train:
            self.train(results)
        self._save_results(results)
        self.print_summary(results)
        return results
