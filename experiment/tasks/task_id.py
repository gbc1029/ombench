from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TaskId:
    subset: str
    case_id: int
    language: str

    def __str__(self) -> str:
        return f"{self.subset}/{self.case_id}/{self.language}"


def parse_task_id(task_id: str) -> Optional[TaskId]:
    if not task_id:
        return None
    parts = task_id.split("/")
    if len(parts) != 3:
        return None
    subset, case_id, language = parts
    try:
        case_id_int = int(case_id)
    except ValueError:
        return None
    return TaskId(subset=subset, case_id=case_id_int, language=language)


def build_task_id(subset: str, case_id: int, language: str) -> str:
    return str(TaskId(subset=subset, case_id=int(case_id), language=language))
