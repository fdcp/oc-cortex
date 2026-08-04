"""验证 P4 查询改写的模板解析和模式互斥规则。"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from code_p4_searcher import (
    _load_query_rewrite_prompt,
    _load_opencode_model_settings,
    _parse_query_rewrite_response,
    _resolve_query_mode,
)


def test_parse_query_rewrite_response_reads_single_query() -> None:
    result = _parse_query_rewrite_response(
        json.dumps({
            "query_type": "implementation",
            "rewritten_query": "序列并行实现与通信流程",
        })
    )

    assert result == ("implementation", "序列并行实现与通信流程")


def test_parse_query_rewrite_response_rejects_missing_query() -> None:
    with pytest.raises(ValueError, match="query_type 或 rewritten_query"):
        _parse_query_rewrite_response('{"query_type":"question"}')


def test_query_rewrite_takes_priority_over_instruction() -> None:
    assert _resolve_query_mode(True, "为这个句子生成表示：") == (True, "")


def test_instruction_is_used_when_query_rewrite_is_disabled() -> None:
    instruction = "为这个句子生成表示："

    assert _resolve_query_mode(False, instruction) == (False, instruction)


def test_original_query_is_used_when_both_modes_are_disabled() -> None:
    assert _resolve_query_mode(False, "") == (False, "")


def test_query_rewrite_prompt_injects_query_without_formatting_json_examples() -> None:
    prompt = _load_query_rewrite_prompt()

    assert "用户查询：{query}" not in prompt
    assert '{"query_type":"implementation"' in prompt


def test_opencode_model_settings_load_endpoint_and_key(tmp_path, monkeypatch) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text('{"opencode-go":{"key":"test-key"}}', encoding="utf-8")
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        """auth_file: {auth_file}
endpoints:
  zen:
    base_url: https://opencode.ai/zen/v1
    auth_provider: opencode-go
models:
  - name: deepseek-v4-flash-free
    endpoint: zen
""".format(auth_file=auth_path),
        encoding="utf-8",
    )

    settings = _load_opencode_model_settings(config_path=config_path)

    assert settings == ("deepseek-v4-flash-free", "https://opencode.ai/zen/v1")
    assert os.environ["OPENCODE_ZEN_API_KEY"] == "test-key"
