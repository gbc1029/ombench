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

import sys

from tqdm import tqdm

# Disable tqdm progress bars when stderr is not a TTY (e.g. nohup, redirected
# output, or after an SSH disconnect) to prevent BrokenPipeError.
_TQDM_DISABLE = not (
    sys.stderr and hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
)

from experiment.client.solver import SolveClient
from experiment.config.settings import SolveSettings
from experiment.prompt.onemillion_prompt import apply_memory, build_plain_prompts
from ombench_eval.evaluator import score_response
from ombench_eval.judge import BaseJudge

logger = logging.getLogger(__name__)


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    message = str(exc).lower()
    return "timeout" in message or "timed out" in message


def _is_gen_timeout(result: Dict[str, Any]) -> bool:
    resp = result.get("response")
    if isinstance(resp, dict) and resp.get("status") == "timeout":
        return True
    return False


def _is_score_timeout(item: Dict[str, Any]) -> bool:
    score = item.get("score")
    if isinstance(score, dict) and score.get("error_type") == "timeout":
        return True
    return False


def _extract_answer(response: Dict[str, Any]) -> str:
    if not isinstance(response, dict):
        return ""
    # Format 1: solve API  →  response["result"]["answer"]
    result = response.get("result")
    if isinstance(result, dict):
        answer = result.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
    # Format 2: OpenAI-style  →  response["choices"][0]["message"]["content"]
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return ""


def _score_ratio(score_info: Any) -> Optional[float]:
    if not isinstance(score_info, dict):
        return None
    max_score = score_info.get("max_score")
    if not max_score:
        return None
    try:
        return float(score_info.get("score", 0)) / float(max_score)
    except Exception:
        return None


SCORE_BUCKET_MID = 0.3
SCORE_BUCKET_HIGH = 0.6
RETRIEVE_THRESHOLD = 0.35
RETRIEVE_K = 3


def _score_bucket(ratio: Optional[float]) -> str:
    if ratio is None:
        return "unknown"
    if ratio >= SCORE_BUCKET_HIGH:
        return "high"
    if ratio >= SCORE_BUCKET_MID:
        return "mid"
    return "low"


def _extract_meta_value(meta: Any, key: str) -> Any:
    if meta is None:
        return None
    model_extra = getattr(meta, "model_extra", None)
    if isinstance(model_extra, dict) and key in model_extra:
        return model_extra.get(key)
    if isinstance(meta, dict):
        return meta.get(key)
    return getattr(meta, key, None)


def _format_memory_sections(memories: List[Dict[str, Any]]) -> Optional[str]:
    if not memories:
        return None

    buckets = {"high": [], "mid": [], "low": [], "unknown": []}
    for mem in memories:
        metadata = mem.get("metadata") if isinstance(mem, dict) else None
        ratio = _extract_meta_value(metadata, "score_ratio")
        if ratio is None:
            success = _extract_meta_value(metadata, "success")
            if success is True:
                bucket = "mid"
            elif success is False:
                bucket = "low"
            else:
                bucket = "unknown"
        else:
            bucket = _score_bucket(float(ratio))
        buckets[bucket].append(mem)

    sections: List[str] = []
    if buckets["high"]:
        sections.append(
            "--- HIGH-SCORE MEMORIES (follow) ---\n"
            + "\n\n".join(
                m.get("content", "") for m in buckets["high"] if isinstance(m, dict)
            )
        )
    if buckets["mid"]:
        sections.append(
            "--- MID-SCORE MEMORIES (use with caution) ---\n"
            + "\n\n".join(
                m.get("content", "") for m in buckets["mid"] if isinstance(m, dict)
            )
        )
    if buckets["low"]:
        sections.append(
            "--- LOW-SCORE FAILURES (avoid) ---\n"
            + "\n\n".join(
                m.get("content", "") for m in buckets["low"] if isinstance(m, dict)
            )
        )
    if buckets["unknown"]:
        sections.append(
            "--- OTHER MEMORIES ---\n"
            + "\n\n".join(
                m.get("content", "") for m in buckets["unknown"] if isinstance(m, dict)
            )
        )

    return "\n\n".join(s for s in sections if s.strip())


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
        self._train_count = 0
        self._train_last_saved = 0
        self._last_checkpoint: Optional[Dict[str, Any]] = None

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
            "error_type": item.get("error_type"),
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
            "error_type": item.get("error_type"),
            "stage": item.get("stage"),
            "system_prompt": item.get("system_prompt"),
            "user_prompt": item.get("user_prompt"),
        }
        if not self.dry_run:
            record["response"] = item.get("response")
        return record

    def _score_item(
        self, item: Dict[str, Any], entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            score = score_response(
                judge=self.judge,
                question=entry.get("question", ""),
                response=item.get("answer", ""),
                rubrics=entry.get("rubrics", []),
                system_prompt=entry.get("system_prompt"),
            )
            if isinstance(score, dict) and score.get("raw") in (
                "single_rubric_object",
                "incomplete_rubric_array",
            ):
                logger.warning(
                    "Task %s score parse incomplete: %s",
                    item.get("task_id"),
                    score.get("raw"),
                )
            item["score"] = score
        except Exception as exc:
            is_timeout = _is_timeout_error(exc)
            if is_timeout:
                logger.warning("Score timeout for task_id: %s", item.get("task_id"))
                item["score"] = {"error_type": "timeout", "raw": str(exc)}
            else:
                logger.error("Task %s score failed: %s", item.get("task_id"), exc)
                item["score"] = {"error_type": "error", "raw": str(exc)}
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
                    retrieval = self.memory_service.retrieve_value_aware(
                        user_prompt, k=RETRIEVE_K, threshold=RETRIEVE_THRESHOLD
                    )
                    selected = (
                        retrieval.get("selected")
                        if isinstance(retrieval, dict)
                        else None
                    )
                    memories: List[Dict[str, Any]] = []
                    if isinstance(selected, dict):
                        memories = [selected]
                    elif isinstance(selected, list):
                        memories = [m for m in selected if isinstance(m, dict)]
                    memory_text = _format_memory_sections(memories)
                    user_prompt = apply_memory(user_prompt, memory_text)
                except Exception as exc:
                    logger.warning("Memory retrieval failed for %s: %s", task_id, exc)
            elif self.memory_context:
                user_prompt = apply_memory(
                    user_prompt, self.memory_context.get(task_id)
                )
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
            payload,
            dry_run=self.dry_run,
            max_retries=self.max_retries,
        )
        result = {
            "task_id": task_id,
            "response": response,
            "answer": _extract_answer(response),
            "system_prompt": prompts["system_prompt"],
            "user_prompt": prompts["user_prompt"],
        }
        if _is_gen_timeout(result):
            result["error"] = "timeout"
            result["error_type"] = "timeout"
            result["stage"] = "generate"
        return result

    def _build_tasks(
        self,
        entries: Dict[str, Dict[str, Any]],
        resume_from: Optional[Path],
    ) -> List[tuple[str, Dict[str, Any]]]:
        skip_ids = _load_completed_ids(resume_from) if resume_from else set()
        if skip_ids:
            logger.info(
                "Resuming: skipping %d already-completed task(s)", len(skip_ids)
            )

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
                tqdm(
                    total=len(future_map),
                    desc=gen_desc,
                    unit="task",
                    position=0,
                    disable=_TQDM_DISABLE,
                ),
            ]
            if total_batches > 1:
                bars.append(
                    tqdm(
                        total=effective_global_total,
                        desc="Overall Generate",
                        unit="task",
                        initial=global_offset,
                        position=1,
                        disable=_TQDM_DISABLE,
                    ),
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
                            "error_type": "timeout"
                            if _is_timeout_error(exc)
                            else "error",
                            "stage": "generate",
                        }
                        results.append(result)
                    self._append_jsonl(
                        self.generated_output, self._format_generated(result)
                    )
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
                tqdm(
                    total=len(tasks),
                    desc=gen_desc,
                    unit="task",
                    position=pos,
                    disable=_TQDM_DISABLE,
                ),
            ]
            pos += 1
            if total_batches > 1:
                bars.append(
                    tqdm(
                        total=effective_global_total,
                        desc="Overall Generate",
                        unit="task",
                        initial=global_offset,
                        position=pos,
                        disable=_TQDM_DISABLE,
                    ),
                )
                pos += 1
            score_bar_idx = len(bars)
            bars.append(
                tqdm(
                    total=len(tasks),
                    desc=score_desc,
                    unit="task",
                    position=pos,
                    disable=_TQDM_DISABLE,
                ),
            )
            pos += 1
            if total_batches > 1:
                bars.append(
                    tqdm(
                        total=effective_global_total,
                        desc="Overall Score",
                        unit="task",
                        initial=global_offset,
                        position=pos,
                        disable=_TQDM_DISABLE,
                    ),
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
                                    "error_type": "timeout"
                                    if _is_timeout_error(exc)
                                    else "error",
                                    "stage": "generate",
                                    "score": None,
                                }

                            self._append_jsonl(
                                self.generated_output, self._format_generated(result)
                            )
                            gen_done += 1
                            bars[0].update(1)
                            if total_batches > 1:
                                bars[1].update(1)

                            if (
                                result.get("answer")
                                and not result.get("error")
                                and not _is_gen_timeout(result)
                            ):
                                entry = task_map.get(tid, {})
                                sf = score_pool.submit(self._score_item, result, entry)
                                score_futures[sf] = tid
                                pending.add(sf)
                            else:
                                result["score"] = None
                                results.append(result)
                                self._append_jsonl(
                                    self.scored_output, self._format_scored(result)
                                )
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
                            self._append_jsonl(
                                self.scored_output, self._format_scored(scored_result)
                            )
                            score_done += 1
                            bars[score_bar_idx].update(1)
                            if total_batches > 1:
                                bars[score_bar_idx + 1].update(1)
            finally:
                for bar in bars:
                    bar.close()

        return results

    def generate_score_train_tasks(
        self,
        tasks: List[tuple[str, Dict[str, Any]]],
        *,
        mode: str = "plain",
        skip_train_ids: Optional[set] = None,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_every: int = 0,
        save_final: bool = True,
        batch_idx: int = 1,
        total_batches: int = 1,
        global_offset: int = 0,
        global_total: int = 0,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        results: List[Dict[str, Any]] = []
        trained_records: List[Dict[str, Any]] = []
        task_map = {task_id: entry for task_id, entry in tasks}
        effective_global_total = global_total or len(tasks)
        skip_train_ids = skip_train_ids or set()

        self._train_count = 0
        self._train_last_saved = 0
        self._last_checkpoint = None

        logger.info(
            "Pipeline: %d tasks, gen_workers=%d, score_workers=%d, mode=%s",
            len(tasks),
            self.workers,
            self.score_workers,
            mode,
        )

        gen_desc = _batch_desc("Generate", batch_idx, total_batches)
        score_desc = _batch_desc("Score", batch_idx, total_batches)
        train_desc = _batch_desc("Train", batch_idx, total_batches)

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
            bars: List[Any] = []

            gen_bar = tqdm(
                total=len(tasks),
                desc=gen_desc,
                unit="task",
                position=pos,
                disable=_TQDM_DISABLE,
            )
            bars.append(gen_bar)
            pos += 1
            gen_overall_bar = None
            if total_batches > 1:
                gen_overall_bar = tqdm(
                    total=effective_global_total,
                    desc="Overall Generate",
                    unit="task",
                    initial=global_offset,
                    position=pos,
                    disable=_TQDM_DISABLE,
                )
                bars.append(gen_overall_bar)
                pos += 1

            score_bar = tqdm(
                total=len(tasks),
                desc=score_desc,
                unit="task",
                position=pos,
                disable=_TQDM_DISABLE,
            )
            bars.append(score_bar)
            pos += 1
            score_overall_bar = None
            if total_batches > 1:
                score_overall_bar = tqdm(
                    total=effective_global_total,
                    desc="Overall Score",
                    unit="task",
                    initial=global_offset,
                    position=pos,
                    disable=_TQDM_DISABLE,
                )
                bars.append(score_overall_bar)
                pos += 1

            train_bar = tqdm(
                total=len(tasks),
                desc=train_desc,
                unit="task",
                position=pos,
                disable=_TQDM_DISABLE,
            )
            bars.append(train_bar)
            pos += 1
            train_overall_bar = None
            if total_batches > 1:
                train_overall_bar = tqdm(
                    total=effective_global_total,
                    desc="Overall Train",
                    unit="task",
                    initial=global_offset,
                    position=pos,
                    disable=_TQDM_DISABLE,
                )
                bars.append(train_overall_bar)

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
                                    "error_type": "timeout"
                                    if _is_timeout_error(exc)
                                    else "error",
                                    "stage": "generate",
                                    "score": None,
                                }

                            self._append_jsonl(
                                self.generated_output, self._format_generated(result)
                            )
                            gen_bar.update(1)
                            if gen_overall_bar:
                                gen_overall_bar.update(1)

                            if (
                                result.get("answer")
                                and not result.get("error")
                                and not _is_gen_timeout(result)
                            ):
                                entry = task_map.get(tid, {})
                                sf = score_pool.submit(self._score_item, result, entry)
                                score_futures[sf] = tid
                                pending.add(sf)
                            else:
                                result["score"] = None
                                results.append(result)
                                self._append_jsonl(
                                    self.scored_output, self._format_scored(result)
                                )
                                score_bar.update(1)
                                if score_overall_bar:
                                    score_overall_bar.update(1)
                                train_bar.update(1)
                                if train_overall_bar:
                                    train_overall_bar.update(1)

                        elif future in score_futures:
                            tid = score_futures[future]
                            try:
                                scored_result = future.result()
                            except Exception as exc:
                                logger.error("Task %s score failed: %s", tid, exc)
                                scored_result = {
                                    "task_id": tid,
                                    "score": {
                                        "error_type": "timeout"
                                        if _is_timeout_error(exc)
                                        else "error",
                                        "raw": str(exc),
                                    },
                                }
                            results.append(scored_result)
                            self._append_jsonl(
                                self.scored_output, self._format_scored(scored_result)
                            )
                            score_bar.update(1)
                            if score_overall_bar:
                                score_overall_bar.update(1)

                            if str(tid) not in skip_train_ids and not _is_score_timeout(
                                scored_result
                            ):
                                record = self._train_one_item(
                                    scored_result,
                                    checkpoint_dir=checkpoint_dir,
                                    checkpoint_every=checkpoint_every,
                                )
                                if record:
                                    trained_records.append(record)

                            train_bar.update(1)
                            if train_overall_bar:
                                train_overall_bar.update(1)

            finally:
                for bar in bars:
                    bar.close()

        if checkpoint_dir and save_final and self._train_count > self._train_last_saved:
            self._last_checkpoint = self._save_checkpoint(checkpoint_dir)

        logger.info(
            "Pipeline complete: %d generated, %d scored, %d trained",
            len(tasks),
            len(results),
            len(trained_records),
        )
        return results, trained_records, self._last_checkpoint

    def score_streaming(
        self,
        results: List[Dict[str, Any]],
        entries: Dict[str, Dict[str, Any]],
        *,
        train: bool = False,
        skip_train_ids: Optional[set] = None,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_every: int = 0,
        save_final: bool = True,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        score_items: List[Dict[str, Any]] = []
        scoreable: List[Dict[str, Any]] = []
        for r in results:
            if r.get("answer") and not r.get("error"):
                entry = entries.get(r["task_id"], {})
                score_items.append(
                    {
                        "item": r,
                        "entry": entry,
                    }
                )
                scoreable.append(r)

        logger.info(
            "Scoring responses: %d / %d tasks, workers=%d",
            len(score_items),
            len(results),
            self.score_workers,
        )

        skip_train_ids = skip_train_ids or set()
        trained_records: List[Dict[str, Any]] = []
        self._train_count = 0
        self._train_last_saved = 0
        self._last_checkpoint = None

        scored_results: List[Dict[str, Any]] = []

        score_bar = tqdm(
            total=len(results),
            desc="Score",
            unit="task",
            position=0,
            disable=_TQDM_DISABLE,
        )
        train_bar = None
        if train:
            train_bar = tqdm(
                total=len(results),
                desc="Train",
                unit="task",
                position=1,
                disable=_TQDM_DISABLE,
            )

        with ThreadPoolExecutor(max_workers=self.score_workers) as pool:
            future_map = {
                pool.submit(self._score_item, item["item"], item["entry"]): item[
                    "item"
                ].get("task_id")
                for item in score_items
            }
            pending = set(future_map.keys())

            try:
                for item in results:
                    if item.get("answer") and not item.get("error"):
                        continue
                    item["score"] = None
                    scored_results.append(item)
                    if self.scored_output:
                        self._append_jsonl(
                            self.scored_output, self._format_scored(item)
                        )
                    score_bar.update(1)
                    if train_bar:
                        train_bar.update(1)

                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        task_id = future_map.get(future)
                        try:
                            scored_result = future.result()
                        except Exception as exc:
                            logger.error("Task %s score failed: %s", task_id, exc)
                            scored_result = {
                                "task_id": task_id,
                                "score": {
                                    "error_type": "timeout"
                                    if _is_timeout_error(exc)
                                    else "error",
                                    "raw": str(exc),
                                },
                            }
                        scored_results.append(scored_result)
                        if self.scored_output:
                            self._append_jsonl(
                                self.scored_output, self._format_scored(scored_result)
                            )
                        score_bar.update(1)

                        if (
                            train
                            and str(task_id) not in skip_train_ids
                            and not _is_score_timeout(scored_result)
                        ):
                            record = self._train_one_item(
                                scored_result,
                                checkpoint_dir=checkpoint_dir,
                                checkpoint_every=checkpoint_every,
                            )
                            if record:
                                trained_records.append(record)
                        if train_bar:
                            train_bar.update(1)
            finally:
                score_bar.close()
                if train_bar:
                    train_bar.close()

        if (
            train
            and checkpoint_dir
            and save_final
            and self._train_count > self._train_last_saved
        ):
            self._last_checkpoint = self._save_checkpoint(checkpoint_dir)

        return scored_results, trained_records, self._last_checkpoint

    def _save_checkpoint(self, checkpoint_dir: Path) -> Optional[Dict[str, Any]]:
        if self.memory_service is None:
            return None
        ckpt_id = time.strftime("%Y%m%d_%H%M%S")
        while ckpt_id in self._checkpoint_ts_used:
            ckpt_id += "_1"
        self._checkpoint_ts_used.add(ckpt_id)
        try:
            return self.memory_service.save_checkpoint_snapshot(
                str(checkpoint_dir), ckpt_id
            )
        except Exception as exc:
            logger.warning("Failed to save checkpoint %s: %s", ckpt_id, exc)
            return None

    def _train_one_item(
        self,
        item: Dict[str, Any],
        *,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_every: int = 0,
    ) -> Optional[Dict[str, Any]]:
        task_id = item.get("task_id")
        if _is_score_timeout(item):
            return None
        if item.get("error") or not item.get("answer"):
            return None
        if self.memory_service is None:
            return None
        response = item.get("response", {})
        status = response.get("status") if isinstance(response, dict) else None
        trajectory = json.dumps(
            response.get("session_data", {}).get("messages", [])
            if isinstance(response, dict)
            else [],
            ensure_ascii=False,
        )
        score_info = item.get("score", {})
        score_ratio = 0.0
        has_ratio = False

        if isinstance(score_info, dict) and score_info.get("max_score"):
            score_ratio = float(score_info.get("score", 0)) / float(
                score_info["max_score"]
            )
            has_ratio = True
        elif status == "completed":
            score_ratio = 1.0
            has_ratio = True

        is_success = status == "completed" and not has_ratio
        if has_ratio:
            is_success = score_ratio >= SCORE_BUCKET_MID

        quality_bucket = _score_bucket(score_ratio if has_ratio else None)

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
                    "score_ratio": score_ratio,
                    "quality_bucket": quality_bucket,
                },
            )
            record = {
                "task_id": task_id,
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._append_jsonl(self.trained_output, record)

            self._train_count += 1
            if checkpoint_dir and checkpoint_every > 0:
                if self._train_count - self._train_last_saved >= checkpoint_every:
                    self._last_checkpoint = self._save_checkpoint(checkpoint_dir)
                    self._train_last_saved = self._train_count
            return record
        except Exception as exc:
            logger.error("Task %s train failed: %s", task_id, exc)
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
            tqdm(
                results, desc=train_desc, unit="task", position=0, disable=_TQDM_DISABLE
            ),
        ]
        if total_batches > 1:
            bars.append(
                tqdm(
                    total=effective_global_total,
                    desc="Overall Train",
                    unit="task",
                    initial=global_offset,
                    position=1,
                    disable=_TQDM_DISABLE,
                ),
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
                if _is_score_timeout(item):
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
                score_ratio = 0.0
                has_ratio = False

                if isinstance(score_info, dict) and score_info.get("max_score"):
                    score_ratio = float(score_info.get("score", 0)) / float(
                        score_info["max_score"]
                    )
                    has_ratio = True
                elif status == "completed":
                    score_ratio = 1.0
                    has_ratio = True

                is_success = status == "completed" and not has_ratio
                if has_ratio:
                    is_success = score_ratio >= SCORE_BUCKET_MID

                quality_bucket = _score_bucket(score_ratio if has_ratio else None)

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
                            "score_ratio": score_ratio,
                            "quality_bucket": quality_bucket,
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

        gen_timeout = 0
        score_timeout = 0
        gen_error = 0
        score_error = 0

        for r in results:
            error_type = r.get("error_type")
            score_info = r.get("score")
            score_error_type = (
                score_info.get("error_type") if isinstance(score_info, dict) else None
            )

            resp = r.get("response")
            resp_status = resp.get("status") if isinstance(resp, dict) else None

            if error_type == "timeout" or resp_status == "timeout":
                gen_timeout += 1
            elif error_type == "error":
                gen_error += 1
            elif score_error_type == "timeout":
                score_timeout += 1
            elif score_error_type == "error":
                score_error += 1
            elif isinstance(score_info, dict) and not score_info.get("max_score"):
                score_error += 1

        scored = [
            r
            for r in results
            if isinstance(r.get("score"), dict) and r["score"].get("max_score")
        ]

        total_fail = gen_timeout + gen_error + score_timeout + score_error

        print(f"\n{'=' * 50}")
        print(f"  Total: {total} | Scored: {len(scored)} | Failed: {total_fail}")

        if scored:
            avg_score = sum(r["score"]["score"] for r in scored) / len(scored)
            avg_max = sum(r["score"]["max_score"] for r in scored) / len(scored)
            pct = (avg_score / avg_max * 100) if avg_max > 0 else 0
            print(f"  Avg Score: {avg_score:.1f}/{avg_max:.1f} ({pct:.1f}%)")
        else:
            print("  Avg Score: N/A (no successful scores)")

        if total_fail:
            print(
                f"  Errors: gen_timeout={gen_timeout}, score_timeout={score_timeout}, "
                f"gen_error={gen_error}, score_error={score_error}"
            )

        print(f"{'=' * 50}\n")

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
