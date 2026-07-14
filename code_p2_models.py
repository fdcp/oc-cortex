"""
Phase 2 数据模型: Task
从 Phase 1 的 Chunk 中提炼出的 session 级任务
"""
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime


@dataclass
class Task:
    """
    一个 session 级任务
    由 Phase 2 LLM 从 session chunks 中提炼而来
    """
    task_id: str              # "{session_id}_T{index}", e.g. "abc_T1"
    session_id: str
    task_label: str           # 简短中文标签 (5-15 字), 如 "修复登录 bug"
    task_summary: str         # 任务总结 (50-200 字)
    chunk_ids: list[str] = field(default_factory=list)  # 关联的 Chunk ID
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(
            task_id=data["task_id"],
            session_id=data["session_id"],
            task_label=data["task_label"],
            task_summary=data["task_summary"],
            chunk_ids=data.get("chunk_ids", []),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )
