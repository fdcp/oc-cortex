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

    def search_entities(self, query: str, limit: int = 20) -> list[dict]:
        """模糊搜索实体名称（LIKE 匹配）"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "SELECT name, entity_type, aliases, source_tasks, task_count "
            "FROM nodes WHERE name LIKE ? ORDER BY task_count DESC LIMIT ?",
            (f"%{query}%", limit),
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
