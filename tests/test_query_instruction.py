import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from code_p1_utils import build_retrieval_query


def test_build_retrieval_query_prepends_instruction() -> None:
    # Given: a retrieval instruction and a user query
    instruction = "为这个句子生成表示以用于检索相关文章："
    query = "GPU对比分析"

    # When: the retrieval query is built
    result = build_retrieval_query(query, instruction)

    # Then: the instruction is directly before the original query
    assert result == "为这个句子生成表示以用于检索相关文章：GPU对比分析"


def test_build_retrieval_query_without_instruction_keeps_query() -> None:
    # Given: no configured retrieval instruction
    query = "GPU对比分析"

    # When: the retrieval query is built
    result = build_retrieval_query(query)

    # Then: the original query is unchanged
    assert result == query
