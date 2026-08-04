import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from code_p3_qdrant_store import Phase3Store, SearchResult


def _store() -> Phase3Store:
    return Phase3Store.__new__(Phase3Store)


def test_hierarchical_rrf_fuses_chunk_paths_before_dense() -> None:
    store = _store()
    dense = [SearchResult("dense-only", 1.0, {"task_id": "dense-only"})]
    summary_sparse = [
        SearchResult("summary-only", 1.0, {"task_id": "summary-only"}),
        SearchResult("shared", 0.5, {"task_id": "shared"}),
    ]
    cleaned_sparse = [
        SearchResult("cleaned-only", 1.0, {"task_id": "cleaned-only"}),
        SearchResult("shared", 0.5, {"task_id": "shared"}),
    ]

    sparse_fused = store._rrf_fuse(summary_sparse, cleaned_sparse, k=1)
    sparse_as_search = [
        SearchResult(result.point_id, result.rrf_score, result.payload)
        for result in sparse_fused
    ]
    fused = store._rrf_fuse(dense, sparse_as_search, k=1)

    assert [result.point_id for result in sparse_fused] == [
        "shared",
        "summary-only",
        "cleaned-only",
    ]
    assert fused[0].point_id == "dense-only"
    assert fused[1].point_id == "shared"
