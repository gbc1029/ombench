from __future__ import annotations

import json
import logging
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
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


def _batch_desc(label: str, batch_idx: int, total_batches: int) -> str:
    if total_batches <= 1:
        return label
    return f"{label} [batch {batch_idx}/{total_batches}]"


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
        generated_output: Optional[Path] = None,
        scored_output: Optional[Path] = None,
        trained_output: Optional[Path] = None,
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
        self._checkpoint_ts_used: set[str] = set()
        self.generated_output = generated_output
        self.scored_output = scored_output
        self.trained_output = trained_output
        self._write_lock = threading.Lock()
        self._initialized_files: set[Path] = set()

    def init_output_files(self, *, append: bool = False) -> None:
        if append:
            return
        for path in (self.generated_output, self.scored_output, self.trained_output):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.open("w", encoding="utf-8").close()
                self._initialized_files.add(path)

    def _append_jsonl(self, path: Optional[Path], record: Dict[str, Any]) -> None:
        if path is None:
            return
        with self._write_lock:
            if path not in self._initialized_files:
                path.parent.mkdir(parents=True, exist_ok=True)
                self._initialized_files.add(path)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _format_generated(self, item: Dict[str, Any]) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "task_id": item.get("task_id"),
            "answer": item.get("answer"),
            "error": item.get("error"),
            "stage": item.get("stage"),
            "system_prompt": item.get("system_prompt"),
            "user_prompt": item.get("user_prompt"),
        }
        if not self.dry_run:
            record["response"] = item.get("response")
        return record

    def _format_scored(self, item: Dict[str, Any]) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "task_id": item.get("task_id"),
            "answer": item.get("answer"),
            "score": item.get("score"),
            "error": item.get("error"),
            "stage": item.get("stage"),
            "system_prompt": item.get("system_prompt"),
            "user_prompt": item.get("user_prompt"),
        }
        if not self.dry_run:
            record["response"] = item.get("response")
        return record

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
            if self.memory_service is not None:
                try:
                    retrieval = self.memory_service.retrieve_value_aware(user_prompt)
                    selected = retrieval.get("selected") if isinstance(retrieval, dict) else None
                    memory_text = None
                    if isinstance(selected, dict):
                        memory_text = selected.get("content")
                    user_prompt = apply_memory(user_prompt, memory_text)
                except Exception as exc:
                    logger.warning("Memory retrieval failed for %s: %s", task_id, exc)
            elif self.memory_context:
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

    def _build_tasks(
        self,
        entries: Dict[str, Dict[str, Any]],
        resume_from: Optional[Path],
    ) -> List[tuple[str, Dict[str, Any]]]:
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
        return tasks

    def generate(
        self,
        entries: Dict[str, Dict[str, Any]],
        *,
        mode: str = "plain",
        resume_from: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        tasks = self._build_tasks(entries, resume_from)
        return self.generate_tasks(tasks, mode=mode)

    def generate_and_score(
        self,
        entries: Dict[str, Dict[str, Any]],
        *,
        mode: str = "plain",
        resume_from: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        tasks = self._build_tasks(entries, resume_from)
        return self.generate_and_score_tasks(tasks, mode=mode)

    def generate_tasks(
        self,
        tasks: List[tuple[str, Dict[str, Any]]],
        *,
        mode: str = "plain",
        batch_idx: int = 1,
        total_batches: int = 1,
        global_offset: int = 0,
        global_total: int = 0,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        effective_global_total = global_total or len(tasks)
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
            gen_desc = _batch_desc("Generate", batch_idx, total_batches)
            bars = [
                tqdm(total=len(future_map), desc=gen_desc, unit="task", position=0),
            ]
            if total_batches > 1:
                bars.append(
                    tqdm(total=effective_global_total, desc="Overall Generate", unit="task", initial=global_offset, position=1),
                )
            try:
                for future in as_completed(future_map):
                    tid = future_map[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as exc:
                        logger.error("Task %s generation failed: %s", tid, exc)
                        result = {
                            "task_id": tid,
                            "response": None,
                            "answer": "",
                            "error": str(exc),
                            "stage": "generate",
                        }
                        results.append(result)
                    self._append_jsonl(self.generated_output, self._format_generated(result))
                    for bar in bars:
                        bar.update(1)
            finally:
                for bar in bars:
                    bar.close()

        return results

    def generate_and_score_tasks(
        self,
        tasks: List[tuple[str, Dict[str, Any]]],
        *,
        mode: str = "plain",
        batch_idx: int = 1,
        total_batches: int = 1,
        global_offset: int = 0,
        global_total: int = 0,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        task_map = {task_id: entry for task_id, entry in tasks}
        effective_global_total = global_total or len(tasks)
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

        gen_desc = _batch_desc("Generate", batch_idx, total_batches)
        score_desc = _batch_desc("Score", batch_idx, total_batches)

        gen_done = 0
        score_done = 0

        with (
            ThreadPoolExecutor(max_workers=self.workers) as gen_pool,
            ThreadPoolExecutor(max_workers=self.score_workers) as score_pool,
        ):
            gen_futures = {
                gen_pool.submit(self._generate_one_direct, tid, ent, mode=mode): tid
                for tid, ent in tasks
            }
            score_futures: Dict[Any, str] = {}

            pos = 0
            bars = [
                tqdm(total=len(tasks), desc=gen_desc, unit="task", position=pos),
            ]
            pos += 1
            if total_batches > 1:
                bars.append(
                    tqdm(total=effective_global_total, desc="Overall Generate", unit="task", initial=global_offset, position=pos),
                )
                pos += 1
            score_bar_idx = len(bars)
            bars.append(
                tqdm(total=len(tasks), desc=score_desc, unit="task", position=pos),
            )
            pos += 1
            if total_batches > 1:
                bars.append(
                    tqdm(total=effective_global_total, desc="Overall Score", unit="task", initial=global_offset, position=pos),
                )

            try:
                pending: set = set(gen_futures.keys())

                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)

                    for future in done:
                        if future in gen_futures:
                            tid = gen_futures[future]
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

                            self._append_jsonl(self.generated_output, self._format_generated(result))
                            gen_done += 1
                            bars[0].update(1)
                            if total_batches > 1:
                                bars[1].update(1)

                            if result.get("answer") and not result.get("error"):
                                entry = task_map.get(tid, {})
                                sf = score_pool.submit(self._score_item, result, entry)
                                score_futures[sf] = tid
                                pending.add(sf)
                            else:
                                result["score"] = None
                                results.append(result)
                                self._append_jsonl(self.scored_output, self._format_scored(result))
                                score_done += 1
                                bars[score_bar_idx].update(1)
                                if total_batches > 1:
                                    bars[score_bar_idx + 1].update(1)

                        elif future in score_futures:
                            tid = score_futures[future]
                            try:
                                scored_result = future.result()
                            except Exception as exc:
                                logger.error("Task %s score failed: %s", tid, exc)
                                scored_result = {
                                    "task_id": tid,
                                    "score": {"raw": str(exc)},
                                }
                            results.append(scored_result)
                            self._append_jsonl(self.scored_output, self._format_scored(scored_result))
                            score_done += 1
                            bars[score_bar_idx].update(1)
                            if total_batches > 1:
                                bars[score_bar_idx + 1].update(1)
            finally:
                for bar in bars:
                    bar.close()

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

    def _save_checkpoint(self, checkpoint_dir: Path) -> Optional[Dict[str, Any]]:
        if self.memory_service is None:
            return None
        ckpt_id = time.strftime("%Y%m%d_%H%M%S")
        while ckpt_id in self._checkpoint_ts_used:
            ckpt_id += "_1"
        self._checkpoint_ts_used.add(ckpt_id)
        try:
            return self.memory_service.save_checkpoint_snapshot(str(checkpoint_dir), ckpt_id)
        except Exception as exc:
            logger.warning("Failed to save checkpoint %s: %s", ckpt_id, exc)
            return None

    def train(
        self,
        results: List[Dict[str, Any]],
        *,
        skip_ids: Optional[set] = None,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_every: int = 0,
        save_final: bool = True,
        batch_idx: int = 1,
        total_batches: int = 1,
        global_offset: int = 0,
        global_total: int = 0,
    ) -> tuple[int, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if self.memory_service is None:
            logger.warning("No memory_service provided, skipping train stage")
            return 0, [], None

        trained = 0
        last_saved = 0
        trained_records: List[Dict[str, Any]] = []
        last_checkpoint: Optional[Dict[str, Any]] = None
        skip_ids = skip_ids or set()
        effective_global_total = global_total or len(results)

        train_desc = _batch_desc("Train", batch_idx, total_batches)
        bars: List[Any] = [
            tqdm(results, desc=train_desc, unit="task", position=0),
        ]
        if total_batches > 1:
            bars.append(
                tqdm(total=effective_global_total, desc="Overall Train", unit="task", initial=global_offset, position=1),
            )

        try:
            for item in bars[0]:
                task_id = item.get("task_id")
                if task_id in skip_ids:
                    if total_batches > 1:
                        bars[1].update(1)
                    continue
                if item.get("error") or not item.get("answer"):
                    if total_batches > 1:
                        bars[1].update(1)
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
                            "task_id": task_id,
                            "request_id": response.get("request_id")
                            if isinstance(response, dict)
                            else None,
                            "task_status": status,
                            "score": score_info,
                        },
                    )
                    trained += 1
                    record = {
                        "task_id": task_id,
                        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    trained_records.append(record)
                    self._append_jsonl(self.trained_output, record)
                    if checkpoint_dir and checkpoint_every > 0:
                        if trained - last_saved >= checkpoint_every:
                            last_checkpoint = self._save_checkpoint(checkpoint_dir)
                            last_saved = trained
                except Exception as exc:
                    logger.error("Task %s train failed: %s", task_id, exc)

                if total_batches > 1:
                    bars[1].update(1)
        finally:
            for bar in bars:
                bar.close()

        if checkpoint_dir and save_final and trained > last_saved:
            last_checkpoint = self._save_checkpoint(checkpoint_dir)

        logger.info("Trained %d / %d tasks", trained, len(results))
        return trained, trained_records, last_checkpoint

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
