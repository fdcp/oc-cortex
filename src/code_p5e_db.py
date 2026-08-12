"""
Phase 5 Extension: SQLite 持久化模块
将知识图谱从 NetworkX gpickle 迁移至 SQLite，支持高效查询和 Graph-RAG 集成。

表结构:
  nodes: 实体节点 (name, entity_type, aliases, source_tasks, task_count)
  edges: 关系边 (head, tail, relation, weight, source_task, extraction_mode)
"""
import json
import sqlite3
from collections import deque
from pathlib import Path

import networkx as nx
from loguru import logger
import contextlib
from typing import List

class KGDatabase:
    """SQLite 知识图谱数据库"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _create_tables(self):
        """创建 nodes 和 edges 表"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                name TEXT PRIMARY KEY,
                entity_type TEXT DEFAULT '',
                aliases TEXT DEFAULT '[]',
                source_tasks TEXT DEFAULT '[]',
                task_count INTEGER DEFAULT 0
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                head TEXT NOT NULL,
                tail TEXT NOT NULL,
                relation TEXT DEFAULT '',
                weight REAL DEFAULT 1.0,
                source_task TEXT DEFAULT '',
                extraction_mode TEXT DEFAULT 'triple',
                FOREIGN KEY (head) REFERENCES nodes(name),
                FOREIGN KEY (tail) REFERENCES nodes(name)
            )
        """)
        # 索引：加速按实体查询和 BFS 扩散
        c.execute("CREATE INDEX IF NOT EXISTS idx_edges_head ON edges(head)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_edges_tail ON edges(tail)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_nodes_task_count ON nodes(task_count DESC)")
        conn.commit()
        conn.close()
        logger.debug(f"KGDatabase 表结构就绪: {self.db_path}")

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def import_graph(self, G: nx.MultiDiGraph, extraction_mode: str = "triple") -> tuple[int, int]:
        """从 NetworkX 图批量导入节点和边到 SQLite。

        Args:
            G: NetworkX MultiDiGraph
            extraction_mode: "triple" 或 "entity"

        Returns:
            (nodes_count, edges_count)
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # 批量写入节点
        nodes_data = []
        for node, attrs in G.nodes(data=True):
            aliases = attrs.get("aliases", [])
            source_tasks = attrs.get("source_tasks", [])
            nodes_data.append((
                node,
                attrs.get("entity_type", ""),
                json.dumps(aliases, ensure_ascii=False),
                json.dumps(source_tasks, ensure_ascii=False),
                len(source_tasks),
            ))
        c.executemany(
            "INSERT OR REPLACE INTO nodes (name, entity_type, aliases, source_tasks, task_count) VALUES (?,?,?,?,?)",
            nodes_data,
        )

        # 批量写入边
        edges_data = []
        for u, v, attrs in G.edges(data=True):
            edges_data.append((
                u,
                v,
                attrs.get("relation", ""),
                attrs.get("weight", 1.0),
                attrs.get("source_task", ""),
                extraction_mode,
            ))
        c.executemany(
            "INSERT INTO edges (head, tail, relation, weight, source_task, extraction_mode) VALUES (?,?,?,?,?,?)",
            edges_data,
        )

        conn.commit()
        conn.close()

        logger.info(f"SQLite 导入完成: {len(nodes_data)} 节点, {len(edges_data)} 边 → {self.db_path}")
        return len(nodes_data), len(edges_data)

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def get_node(self, name: str) -> dict | None:
        """按名称查询节点"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT name, entity_type, aliases, source_tasks, task_count FROM nodes WHERE name = ?", (name,))
        row = c.fetchone()
        conn.close()
        if not row:
            return None
        return {
            "name": row[0],
            "entity_type": row[1],
            "aliases": json.loads(row[2]),
            "source_tasks": json.loads(row[3]),
            "task_count": row[4],
        }

    def get_edges(self, name: str, direction: str = "both") -> list[dict]:
        """查询与指定节点关联的边。

        Args:
            name: 节点名称
            direction: "out" (head=name), "in" (tail=name), "both"
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        results = []

        if direction in ("out", "both"):
            c.execute(
                "SELECT head, tail, relation, weight, source_task, extraction_mode FROM edges WHERE head = ?",
                (name,),
            )
            for row in c.fetchall():
                results.append({
                    "head": row[0], "tail": row[1], "relation": row[2],
                    "weight": row[3], "source_task": row[4], "extraction_mode": row[5],
                })

        if direction in ("in", "both"):
            c.execute(
                "SELECT head, tail, relation, weight, source_task, extraction_mode FROM edges WHERE tail = ?",
                (name,),
            )
            for row in c.fetchall():
                results.append({
                    "head": row[0], "tail": row[1], "relation": row[2],
                    "weight": row[3], "source_task": row[4], "extraction_mode": row[5],
                })

        conn.close()
        return results

    def get_tasks_by_entity(self, name: str) -> list[str]:
        """查询指定实体关联的所有 task_id"""
        node = self.get_node(name)
        if not node:
            return []
        return node["source_tasks"]

    def search_entities(self, query: str, limit: int = 20) -> List[dict]:
        """
        Fuzzy-search entity: match name OR any alias in aliases array.
        Support searching only by alias even if name does not contain query.
        Use GROUP BY + MIN(score) instead of DISTINCT to preserve best match rank.
        Sort priority:
            0: exact match on name
            1: prefix match on name
            2: exact match on any alias
            3: prefix match on any alias
            4: partial contains match on name
            5: partial contains match on any alias
        Within same rank: shorter name first, higher task_count first.
        """
        query = query.strip()
        if not isinstance(query, str) or not query:
            return []
        limit = max(1, min(int(limit), 200))

        # Escape LIKE special characters: \ % _
        esc = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pat_contains = f"%{esc}%"
        pat_prefix = f"{esc}%"

        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("PRAGMA case_sensitive_like = false;")
            c = conn.cursor()

            c.execute(
                """
                SELECT
                    n.name,
                    n.entity_type,
                    n.aliases,
                    n.source_tasks,
                    n.task_count
                FROM nodes n
                LEFT JOIN json_each(NULLIF(n.aliases, '')) al
                WHERE
                    n.name LIKE ? ESCAPE '\\'
                    OR al.value LIKE ? ESCAPE '\\'
                GROUP BY n.rowid, n.name
                ORDER BY
                    MIN(
                        CASE
                            WHEN n.name = ?                    THEN 0
                            WHEN n.name LIKE ? ESCAPE '\\'     THEN 1
                            WHEN al.value = ?                  THEN 2
                            WHEN al.value LIKE ? ESCAPE '\\'   THEN 3
                            WHEN n.name LIKE ? ESCAPE '\\'     THEN 4
                            ELSE                                    5
                        END
                    ),
                    LENGTH(n.name) ASC,
                    n.task_count DESC
                LIMIT ?
                """,
                (
                    pat_contains, pat_contains,
                    query, pat_prefix,
                    query, pat_prefix,
                    pat_contains,
                    limit,
                ),
            )
            rows = c.fetchall()

        def _safe_load_json(s, default):
            try:
                return json.loads(s) if s else default
            except (json.JSONDecodeError, TypeError):
                return default

        return [
            {
                "name": r[0],
                "entity_type": r[1],
                "aliases": _safe_load_json(r[2], []),
                "source_tasks": _safe_load_json(r[3], []),
                "task_count": r[4],
            }
            for r in rows
        ]

    def bfs_expand(self, seed_entities: list[str], depth: int = 1, max_nodes: int = 50) -> set[str]:
        """从种子实体出发做 BFS 扩散，返回扩散到的所有实体名。

        Args:
            seed_entities: 种子实体列表
            depth: BFS 扩散深度 (默认 1 跳)
            max_nodes: 最大扩散节点数 (防止爆炸)
        """
        visited = set(seed_entities)
        queue = deque()
        for e in seed_entities:
            queue.append((e, 0))

        while queue and len(visited) < max_nodes:
            current, d = queue.popleft()
            if d >= depth:
                continue
            edges = self.get_edges(current, direction="both")
            for edge in edges:
                neighbor = edge["tail"] if edge["head"] == current else edge["head"]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, d + 1))
                    if len(visited) >= max_nodes:
                        break

        return visited

    def get_top_entities(self, limit: int = 20) -> list[dict]:
        """获取关联 task 最多的 Top 实体"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "SELECT name, entity_type, aliases, source_tasks, task_count "
            "FROM nodes ORDER BY task_count DESC LIMIT ?",
            (limit,),
        )
        results = []
        for row in c.fetchall():
            results.append({
                "name": row[0],
                "entity_type": row[1],
                "aliases": json.loads(row[2]),
                "source_tasks": json.loads(row[3]),
                "task_count": row[4],
            })
        conn.close()
        return results

    def get_stats(self) -> dict:
        """数据库统计信息"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM nodes")
        node_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM edges")
        edge_count = c.fetchone()[0]
        c.execute("SELECT COUNT(DISTINCT extraction_mode) FROM edges")
        mode_count = c.fetchone()[0]
        c.execute("SELECT AVG(task_count) FROM nodes")
        avg_task_count = c.fetchone()[0] or 0
        conn.close()
        return {
            "db_path": self.db_path,
            "nodes": node_count,
            "edges": edge_count,
            "modes": mode_count,
            "avg_task_count_per_entity": round(avg_task_count, 2),
        }
