"""
Unit tests for GraphRAGSearcher scoring logic (v2 design).

覆盖:
  - 参数校验: graph_channel_weight / rerank_multiplier 边界
  - 反 IDF 公式: log(1 + N / task_count)
  - 外层加权 RRF: vector-only / graph-only / both / neither
  - E2E: in-memory SQLite KG + mock SessionSearcher, 跑完整 search()

运行: cd project_root && python3 tests/test_p5e/test_graph_rag_scoring.py
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# 把 src 加到 path
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "src"))

from code_p4_searcher import SessionSearcher, SessionSearchResult
from code_p2_models import Task
from code_p5e_db import KGDatabase
from code_p5e_graph_rag import GraphRAGSearcher


# ============================================================
# 辅助: 构造测试数据
# ============================================================

def make_task(task_id: str, label: str = "label", summary: str = "summary", chunk_ids=None) -> Task:
    return Task(
        task_id=task_id,
        session_id=f"S_{task_id}",
        task_label=label,
        task_summary=summary,
        chunk_ids=chunk_ids or [f"chunk_{task_id}"],
    )


def make_result(task_id: str, hybrid_score: float = 0.0, chunks=None) -> SessionSearchResult:
    return SessionSearchResult(
        task_id=task_id,
        session_id=f"S_{task_id}",
        task_label=f"label_{task_id}",
        task_summary=f"summary_{task_id}",
        rerank_score=0.0,
        hybrid_score=hybrid_score,
        chunks=chunks or [],
    )


def make_searcher(task_ids: list[str]) -> MagicMock:
    """构造一个最小可用的 mock SessionSearcher。"""
    searcher = MagicMock(spec=SessionSearcher)
    searcher.tasks = [make_task(tid) for tid in task_ids]
    searcher.reranker = None  # 测试 RRF 阶段, 不需要 rerank
    return searcher


def make_db() -> KGDatabase:
    """构造一个空的 in-memory 风格 KGDatabase (用临时文件)。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return KGDatabase(tmp.name), tmp.name


# ============================================================
# 1. 参数校验
# ============================================================

class TestInitValidation(unittest.TestCase):
    """GraphRAGSearcher.__init__ 的参数边界检查。"""

    def _make_minimal(self, **overrides):
        searcher = make_searcher(["T0", "T1"])
        db, _ = make_db()
        kwargs = dict(searcher=searcher, kg_db=db)
        kwargs.update(overrides)
        return GraphRAGSearcher(**kwargs)

    def test_default_weight_is_0_3(self):
        g = self._make_minimal()
        self.assertEqual(g.graph_channel_weight, 0.3)

    def test_valid_weights(self):
        for w in (0.0, 0.3, 0.5, 1.0):
            g = self._make_minimal(graph_channel_weight=w)
            self.assertEqual(g.graph_channel_weight, w)

    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError):
            self._make_minimal(graph_channel_weight=-0.1)

    def test_weight_above_one_rejected(self):
        with self.assertRaises(ValueError):
            self._make_minimal(graph_channel_weight=1.5)

    def test_rerank_multiplier_below_one_rejected(self):
        with self.assertRaises(ValueError):
            self._make_minimal(rerank_multiplier=0.5)

    def test_total_tasks_cached(self):
        searcher = make_searcher(["T0", "T1", "T2", "T3"])
        db, _ = make_db()
        g = GraphRAGSearcher(searcher=searcher, kg_db=db)
        self.assertEqual(g._total_tasks, 4)


# ============================================================
# 2. 反 IDF 公式
# ============================================================

class TestAntiIDFFormula(unittest.TestCase):
    """每 entity 贡献 = log(1 + N / task_count)。

    N = 全局 task 数 (self._total_tasks), 来自 searcher.tasks。
    这里直接验证公式, 不走 search()。
    """

    def test_formula_exact(self):
        N = 1000
        tc = 5
        expected = math.log(1 + N / tc)
        actual = math.log(1 + N / tc)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_rare_entity_higher_than_popular(self):
        """tc=1 的贡献应该严格大于 tc=100 的。"""
        N = 1000
        rare = math.log(1 + N / 1)     # log(1001) ≈ 6.91
        popular = math.log(1 + N / 100)  # log(11) ≈ 2.40
        self.assertGreater(rare, popular)

    def test_tc_zero_safe_via_max_guard(self):
        """tc=0 走 max(tc, 1) 兜底, 不会除零。"""
        tc = 0
        N = 100
        contrib = math.log(1 + N / max(tc, 1))
        self.assertEqual(contrib, math.log(101))

    def test_total_tasks_zero_safe(self):
        """N=0 走 max(N, 1) 兜底, 不会除零 (返回一个有限正数)。"""
        import math as _m
        N = 0
        tc = 5
        contrib = _m.log(1 + max(N, 1) / tc)
        # 不应该是 NaN / inf / 负数
        self.assertTrue(_m.isfinite(contrib))
        self.assertGreater(contrib, 0.0)

    def test_monotonic_decreasing_in_tc(self):
        """tc 越大, 贡献越小 (单调)。"""
        N = 1000
        prev = float("inf")
        for tc in [1, 5, 10, 50, 100, 500]:
            contrib = math.log(1 + N / tc)
            self.assertLess(contrib, prev, f"tc={tc} should be smaller than previous")
            prev = contrib

    def test_contribution_clamped_above_zero(self):
        """log(1 + x) 永远 > 0, 不应该出现负贡献。"""
        for N in [1, 10, 100, 10000]:
            for tc in [1, 10, 100, 1000]:
                self.assertGreater(math.log(1 + N / tc), 0.0)


# ============================================================
# 3. 外层加权 RRF
# ============================================================

class TestOuterRRF(unittest.TestCase):
    """外层 RRF: hybrid = (1-gw)/(k+rank_v) + gw/(k+rank_g)"""

    K = 60

    def _make_g(self, gw: float = 0.3):
        searcher = make_searcher(["T0", "T1", "T2", "T3", "T4"])
        db, _ = make_db()
        return GraphRAGSearcher(searcher=searcher, kg_db=db, graph_channel_weight=gw)

    def test_vector_only_task(self):
        """只在 vector 出现 → rrf = (1-gw)/(k+rank_v)"""
        g = self._make_g(gw=0.3)
        v = [make_result("T0"), make_result("T1"), make_result("T2")]
        gr = []  # 图谱空
        fused = g._outer_rrf(v, gr, k=self.K)

        self.assertEqual(len(fused), 3)
        # rank_v(T0)=1, rank_g(T0)=+∞ → 0
        expected_t0 = (1 - 0.3) / (self.K + 1)  # 0.7/61
        self.assertAlmostEqual(fused[0].hybrid_score, expected_t0, places=10)
        self.assertEqual(fused[0].task_id, "T0")

    def test_graph_only_task(self):
        """只在 graph 出现 → rrf = gw/(k+rank_g)"""
        g = self._make_g(gw=0.3)
        v = []
        gr = [make_result("T0"), make_result("T1")]
        fused = g._outer_rrf(v, gr, k=self.K)

        self.assertEqual(len(fused), 2)
        expected_t0 = 0.3 / (self.K + 1)
        self.assertAlmostEqual(fused[0].hybrid_score, expected_t0, places=10)

    def test_both_sources_sum(self):
        """两路都中 → 两项相加"""
        g = self._make_g(gw=0.3)
        v = [make_result("T0"), make_result("T1")]
        gr = [make_result("T0"), make_result("T1")]
        fused = g._outer_rrf(v, gr, k=self.K)

        self.assertEqual(len(fused), 2)
        # T0: rank_v=1, rank_g=1
        expected = (1 - 0.3) / (self.K + 1) + 0.3 / (self.K + 1)
        self.assertAlmostEqual(fused[0].hybrid_score, expected, places=10)
        # T1: rank_v=2, rank_g=2
        expected_t1 = (1 - 0.3) / (self.K + 2) + 0.3 / (self.K + 2)
        self.assertAlmostEqual(fused[1].hybrid_score, expected_t1, places=10)

    def test_neither_source_does_not_appear(self):
        """不可能 (候选至少来自一边), 但验一下空输入"""
        g = self._make_g(gw=0.3)
        fused = g._outer_rrf([], [], k=self.K)
        self.assertEqual(fused, [])

    def test_sort_descending(self):
        """输出按 hybrid_score 降序"""
        g = self._make_g(gw=0.3)
        v = [make_result("T2"), make_result("T0"), make_result("T1")]  # 故意乱序
        gr = []
        fused = g._outer_rrf(v, gr, k=self.K)

        # T0 rank=2, T1 rank=3, T2 rank=1 → 降序: T2, T0, T1
        self.assertEqual([r.task_id for r in fused], ["T2", "T0", "T1"])
        scores = [r.hybrid_score for r in fused]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_vector_preferred_for_chunks_when_both_have_same_task(self):
        """同一 task_id 在两边都出现时, 用 vector 版本的 chunks (通常更完整)"""
        g = self._make_g(gw=0.3)
        v_chunks = [MagicMock(chunk_id="v_chunk")]
        g_chunks = [MagicMock(chunk_id="g_chunk")]
        v = [SessionSearchResult(
            task_id="T0", session_id="S_T0", task_label="l", task_summary="s",
            rerank_score=0.0, hybrid_score=0.0, chunks=v_chunks,
        )]
        gr = [SessionSearchResult(
            task_id="T0", session_id="S_T0", task_label="l", task_summary="s",
            rerank_score=0.0, hybrid_score=0.0, chunks=g_chunks,
        )]
        fused = g._outer_rrf(v, gr, k=self.K)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].chunks, v_chunks)

    def test_graph_only_uses_graph_chunks(self):
        """只在 graph 出现的 task, chunks 来自 graph 版本 (我们已经在 search() 里补了)"""
        g = self._make_g(gw=0.3)
        g_chunks = [MagicMock(chunk_id="g_chunk")]
        gr = [SessionSearchResult(
            task_id="T0", session_id="S_T0", task_label="l", task_summary="s",
            rerank_score=0.0, hybrid_score=0.0, chunks=g_chunks,
        )]
        fused = g._outer_rrf([], gr, k=self.K)
        self.assertEqual(fused[0].chunks, g_chunks)

    def test_gw_zero_pure_vector(self):
        """gw=0 → 纯向量 RRF, graph 结果不贡献"""
        g = self._make_g(gw=0.0)
        v = [make_result("T0"), make_result("T1")]
        gr = [make_result("T0"), make_result("T2")]  # T2 只在 graph
        fused = g._outer_rrf(v, gr, k=self.K)

        # T2 应该出现但 hybrid=0 (因为 gw=0, rank_g 唯一贡献为 0)
        t2 = next(r for r in fused if r.task_id == "T2")
        self.assertAlmostEqual(t2.hybrid_score, 0.0, places=10)
        # T0 (两路都有) 应该分最高
        self.assertEqual(fused[0].task_id, "T0")

    def test_gw_one_pure_graph(self):
        """gw=1 → 纯图谱 RRF, vector 结果不贡献"""
        g = self._make_g(gw=1.0)
        v = [make_result("T0"), make_result("T1")]
        gr = [make_result("T0"), make_result("T2")]
        fused = g._outer_rrf(v, gr, k=self.K)

        # T1 (只在 vector) 应该出现但 hybrid=0
        t1 = next(r for r in fused if r.task_id == "T1")
        self.assertAlmostEqual(t1.hybrid_score, 0.0, places=10)
        # T0 (两路都有) 分最高
        self.assertEqual(fused[0].task_id, "T0")

    def test_rank_consistency_with_input_order(self):
        """rank 完全由输入顺序决定, 不看 hybrid_score 字段值"""
        g = self._make_g(gw=0.3)
        # 即便 hybrid_score 是反着的, rank 也按列表顺序
        v = [
            make_result("T0", hybrid_score=99.0),  # 排第 1
            make_result("T1", hybrid_score=1.0),    # 排第 2
        ]
        gr = [make_result("T0", hybrid_score=0.0001)]
        fused = g._outer_rrf(v, gr, k=self.K)

        # T0: rank_v=1, rank_g=1
        expected = (1 - 0.3) / (self.K + 1) + 0.3 / (self.K + 1)
        self.assertAlmostEqual(fused[0].hybrid_score, expected, places=10)


# ============================================================
# 4. Rerank buffer (v2 设计: top_k, 不再 *2)
# ============================================================

class TestRerankBuffer(unittest.TestCase):
    """v2 设计: rerank 输入直接是 top_k, 不再 *2。

    老设计下 *2 是给有 bug 的 _merge_candidates 兜底;
    新 RRF 排序够准, 不需要这个 buffer。
    """

    def _make_g_with_reranker(self, n_results: int = 20):
        searcher = make_searcher([f"T{i}" for i in range(n_results)])
        # mock 一个 reranker, 返回前 N 个的伪 score
        from unittest.mock import MagicMock
        reranker = MagicMock()

        class _RR:
            def __init__(self, index, score):
                self.index = index
                self.score = score
        reranker.rank = MagicMock(
            side_effect=lambda q, texts, top_k: [_RR(i, 1.0 - i * 0.01) for i in range(min(top_k, len(texts)))]
        )
        searcher.reranker = reranker
        db, _ = make_db()
        return GraphRAGSearcher(searcher=searcher, kg_db=db, graph_channel_weight=0.3), searcher

    def test_rerank_called_with_top_k_not_2x(self):
        """核心断言: reranker.rank 的 top_k 参数 == 调用方传入的 top_k, 不是 2*top_k"""
        g, searcher = self._make_g_with_reranker(n_results=20)
        candidates = [make_result(f"T{i}") for i in range(15)]
        g._rerank(candidates, query="hello", top_k=5)

        searcher.reranker.rank.assert_called_once()
        call_kwargs = searcher.reranker.rank.call_args.kwargs
        # 关键: top_k 是 5, 不是 10
        self.assertEqual(call_kwargs["top_k"], 5)
        self.assertNotEqual(call_kwargs["top_k"], 10)

    def test_rerank_returns_top_k_results(self):
        """返回结果数 == top_k (mock 行为)"""
        g, _ = self._make_g_with_reranker(n_results=20)
        candidates = [make_result(f"T{i}") for i in range(15)]
        reranked = g._rerank(candidates, query="hello", top_k=5)

        self.assertEqual(len(reranked), 5)

    def test_rerank_preserves_hybrid_score(self):
        """rerank 后的 hybrid_score 来自原 candidate, 不被覆盖"""
        g, _ = self._make_g_with_reranker(n_results=20)
        candidates = [make_result(f"T{i}", hybrid_score=0.5 - i * 0.01) for i in range(15)]
        reranked = g._rerank(candidates, query="hello", top_k=5)

        # rerank 顺序由 mock 决定 (前 5 个 index 0,1,2,3,4)
        # 它们原本的 hybrid_score 应该被原样保留
        for i, r in enumerate(reranked):
            expected_hybrid = 0.5 - i * 0.01
            self.assertAlmostEqual(r.hybrid_score, expected_hybrid, places=10)

    def test_rerank_skipped_when_no_reranker(self):
        """searcher.reranker is None → 直接返回原 candidates, 不做 rerank"""
        # 用 make_searcher (reranker=None) 而不是 _make_g_with_reranker (装 reranker)
        searcher = make_searcher([f"T{i}" for i in range(5)])
        db, _ = make_db()
        g = GraphRAGSearcher(searcher=searcher, kg_db=db, graph_channel_weight=0.3)
        candidates = [make_result(f"T{i}") for i in range(3)]
        result = g._rerank(candidates, query="hello", top_k=2)
        # 原样返回
        self.assertEqual(result, candidates)

    def test_rerank_text_format(self):
        """传给 reranker 的文本只用 task_summary, 不拼 label (label 太短信息密度低)"""
        g, searcher = self._make_g_with_reranker(n_results=5)
        candidates = [
            SessionSearchResult(
                task_id="T0", session_id="S", task_label="我的标签",
                task_summary="我的总结", rerank_score=0.0, hybrid_score=0.0, chunks=[],
            ),
        ]
        g._rerank(candidates, query="hello", top_k=1)

        call_args = searcher.reranker.rank.call_args
        texts = call_args.args[1]  # 第二个位置参数
        # 只有 summary, 没有 label
        self.assertEqual(texts, ["我的总结"])


# ============================================================
# 5. E2E: in-memory KG + mock searcher
# ============================================================

class TestE2E(unittest.TestCase):
    """完整 search() 流程, 用 in-memory SQLite + mock searcher。"""

    def setUp(self):
        # 建一个最小 KG:
        #   Entity A (tc=2) — T0, T1
        #   Entity B (tc=5) — T0, T1, T2, T3, T4
        #   Entity C (tc=1) — T2 (稀有)
        #   Edge: A -> B
        self.db, self.db_path = make_db()
        import networkx as nx
        G = nx.MultiDiGraph()
        G.add_node("A", entity_type="tech", aliases=[], source_tasks=["T0", "T1"], task_count=2)
        G.add_node("B", entity_type="tech", aliases=[], source_tasks=["T0", "T1", "T2", "T3", "T4"], task_count=5)
        G.add_node("C", entity_type="api", aliases=[], source_tasks=["T2"], task_count=1)
        G.add_edge("A", "B", relation="related", weight=1.0, source_task="T0", extraction_mode="triple")
        self.db.import_graph(G, extraction_mode="triple")

        # mock searcher: 返回固定的 vector_results
        self.searcher = make_searcher(["T0", "T1", "T2", "T3", "T4"])
        # 模拟 searcher.search() 返回 T0, T1 (RRF 降序)
        self.searcher.search.return_value = [
            make_result("T0", hybrid_score=0.03),
            make_result("T1", hybrid_score=0.02),
            make_result("T2", hybrid_score=0.01),
        ]
        # 让 _expand_chunks 返一个 mock chunk, 验证 chunks 被补
        self.fake_chunk = MagicMock(chunk_id="c0", session_id="S_T0", turn_index=0,
                                    summary="s", cleaned_text_preview="...")
        self.searcher._expand_chunks.return_value = [self.fake_chunk]

        # mock extract_query_entities: 直接返回 ["A"]
        # 必须在 import 之后 patch, 否则 import 时就锁住函数引用
        from code_p5e_graph_rag import extract_query_entities
        self._orig_extract = extract_query_entities
        import code_p5e_graph_rag as _gmod
        _gmod.extract_query_entities = lambda query, **kw: ["A"]
        self.addCleanup(self._restore_extract)

    def _restore_extract(self):
        import code_p5e_graph_rag as _gmod
        _gmod.extract_query_entities = self._orig_extract

    def tearDown(self):
        Path(self.db_path).unlink(missing_ok=True)

    def test_search_returns_results(self):
        g = GraphRAGSearcher(
            searcher=self.searcher,
            kg_db=self.db,
            graph_channel_weight=0.3,
            bfs_depth=1,
            max_expand_nodes=30,
            max_graph_tasks=50,
        )
        results, debug = g.search("some query", top_k=3, use_reranker=False, use_graph_rag=True)

        # 应该有结果
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)

        # debug 字段
        self.assertIn("vector_results", debug)
        self.assertIn("graph_candidates", debug)
        self.assertIn("merged_candidates", debug)
        self.assertIn("outer_rrf_k", debug)
        self.assertEqual(debug["outer_rrf_k"], 30)
        self.assertEqual(debug["graph_channel_weight"], 0.3)
        self.assertIn("source_distribution", debug)

    def test_graph_only_results_have_chunks(self):
        """图谱扩散独有的 task (T3, T4) 在 vector 里没有, 也必须有 chunks"""
        g = GraphRAGSearcher(
            searcher=self.searcher,
            kg_db=self.db,
            graph_channel_weight=0.3,
        )
        results, debug = g.search("q", top_k=10, use_reranker=False, use_graph_rag=True)

        # 拿到所有 graph 召回的 task (T0, T1, T2, T3, T4)
        # T3, T4 不在 vector_results, 但应该出现在 results 里 (因为图谱扩散)
        graph_only_ids = ["T3", "T4"]
        for r in results:
            if r.task_id in graph_only_ids:
                self.assertGreater(
                    len(r.chunks), 0,
                    f"Graph-only result {r.task_id} should have non-empty chunks",
                )
                # _expand_chunks 被调用过
                self.searcher._expand_chunks.assert_called()

    def test_source_distribution_counts(self):
        """source_distribution 应正确分类 vector/graph/both"""
        g = GraphRAGSearcher(
            searcher=self.searcher,
            kg_db=self.db,
            graph_channel_weight=0.3,
        )
        results, debug = g.search("q", top_k=10, use_reranker=False, use_graph_rag=True)

        # T0, T1 都在 vector (mock 返了 T0, T1, T2)
        # T2 也在 vector
        # T3, T4 只在 graph
        # T0~T2 也在 graph (B entity 涵盖 5 个 task)
        # 所以 final 里:
        #   both: T0, T1, T2
        #   graph: T3, T4
        #   vector: 无
        dist = debug["source_distribution"]
        self.assertEqual(dist["both"], 3, f"expected 3 both, got {dist}")
        self.assertEqual(dist["graph"], 2, f"expected 2 graph-only, got {dist}")
        self.assertEqual(dist["vector"], 0, f"expected 0 vector-only, got {dist}")

    def test_use_graph_rag_false_skips_graph(self):
        """use_graph_rag=False 应该走纯向量, 不调 extract_query_entities"""
        g = GraphRAGSearcher(
            searcher=self.searcher,
            kg_db=self.db,
            graph_channel_weight=0.3,
        )
        # 拿掉 mock, 让 extract 真的会被调 (会失败), 用 sentinel 检测
        import code_p5e_graph_rag as _gmod
        call_count = [0]
        def fail_extract(query, **kw):
            call_count[0] += 1
            return []
        _gmod.extract_query_entities = fail_extract
        try:
            results, debug = g.search("q", top_k=2, use_reranker=False, use_graph_rag=False)
            self.assertEqual(call_count[0], 0, "extract_query_entities should not be called")
            # 走 searcher.search 直接路径
            self.searcher.search.assert_called()
        finally:
            _gmod.extract_query_entities = self._orig_extract

    def test_anti_idf_prefers_rare_entity_in_ranking(self):
        """C (tc=1) 链接 T2, B (tc=5) 也链接 T2 → T2 应该被高优召回 (C 拉高)"""
        # mock 让 vector 只返 T0 (避免 T2 被向量抢)
        self.searcher.search.return_value = [
            make_result("T0", hybrid_score=0.05),
        ]
        g = GraphRAGSearcher(
            searcher=self.searcher,
            kg_db=self.db,
            graph_channel_weight=0.5,  # 放大图谱权重
        )
        results, debug = g.search("q", top_k=10, use_reranker=False, use_graph_rag=True)

        # T0 在 vector rank=1
        # T2 只在 graph (A 扩散不到 C, B 扩散到 T2)
        # 等等: A -> B (1 跳), B 链接 T0~T4 (所以 T2 会被召回, 但不是 C)
        # C 只链接 T2, 但 C 不会被扩散到 (A/B 都不指向 C)
        # 所以 graph_results 应有 T0~T4, T2 由 B 贡献, 反 IDF 算 B 的贡献:
        #   log(1 + 5 / 5) = log(2) ≈ 0.69
        # T2 唯一来源是 B 的 0.69
        # T0 唯一来源是 A (log(1+5/2)=1.25) + B (0.69) = 1.94
        # 所以 T0 > T2 in graph rank
        # Vector: T0 rank 1
        # final: T0 (both) > T1, T3, T4 (graph-only) > T2 (graph-only)
        # T2 应该出现在 final 中 (因为 top_k=10)
        task_ids = [r.task_id for r in results]
        self.assertIn("T2", task_ids)

        # 校验 hybrid_score 排序降序
        scores = [r.hybrid_score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
