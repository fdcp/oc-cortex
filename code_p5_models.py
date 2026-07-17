"""
Phase 5 数据模型: Triple, Entity, KGStats
知识图谱的三元组、实体和统计信息
"""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Triple:
    """知识图谱三元组"""
    head: str               # 头实体
    relation: str           # 关系 (动词短语)
    tail: str               # 尾实体
    confidence: float       # 置信度 (0-1)
    source_task_id: str     # 来源 task ID

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Triple":
        return cls(
            head=data["head"],
            relation=data["relation"],
            tail=data["tail"],
            confidence=data.get("confidence", 0.5),
            source_task_id=data.get("source_task_id", ""),
        )


@dataclass
class Entity:
    """知识图谱实体"""
    name: str                                      # 原始实体名
    entity_type: str = "unknown"                   # person/project/module/file/concept/bug/tool
    canonical_name: str = ""                       # 对齐后的标准名 (初始为空, 对齐时填充)
    source_tasks: list[str] = field(default_factory=list)  # 出现的 task_id 列表

    def __post_init__(self):
        if not self.canonical_name:
            self.canonical_name = self.name

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        return cls(
            name=data["name"],
            entity_type=data.get("entity_type", "unknown"),
            canonical_name=data.get("canonical_name", data["name"]),
            source_tasks=data.get("source_tasks", []),
        )


@dataclass
class KGStats:
    """知识图谱统计"""
    total_tasks: int = 0          # 处理的 task 总数
    total_triples: int = 0        # 抽取的三元组总数
    total_entities: int = 0       # 去重前的实体总数 (每个三元组贡献 head+tail)
    unique_entities: int = 0      # 对齐后的唯一实体数
    merged_pairs: int = 0         # 合并的实体对数
    total_edges: int = 0          # 图谱中的边数 (MultiDiGraph 可能有重边)
    total_nodes: int = 0          # 图谱中的节点数
