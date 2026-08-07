"""用途：从 OpenCode Go/Zen 与 MiniMax 文档生成模型池配置。

使用方式：在项目根目录运行 `python3 src/code_update_opencode_models.py`，
可通过 `--output`、`--go-url`、`--zen-url` 和 `--minimax-url` 覆盖默认路径或文档地址。

输入：Go 文档的模型端点表、Zen 文档的模型端点表与定价表、MiniMax API 概览页的语言模型表。
期望输出：`config/opencode_models.yaml`，包含全部 Go 模型、免费的 Zen 模型和 MiniMax 语言模型
（M3 / M2.x 全系，H3 视频模型不在范围内）。
"""

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import argparse
import ssl
import sys
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi
import yaml


GO_DOC_URL: Final = "https://opencode.ai/docs/zh-cn/go/"
ZEN_DOC_URL: Final = "https://opencode.ai/docs/zh-cn/zen/"
MINIMAX_DOC_URL: Final = "https://platform.minimaxi.com/docs/api-reference/api-overview"
DEFAULT_OUTPUT: Final = Path(__file__).parents[1] / "config" / "opencode_models.yaml"
USER_AGENT: Final = "oc-sess-graph/model-pool-updater"


@dataclass(frozen=True, slots=True)
class ModelRecord:
    name: str
    endpoint: str
    label: str = ""


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.section = ""
        self._heading: list[str] = []
        self._cell: list[str] = []
        self._row: list[str] = []
        self._table: list[tuple[str, list[list[str]]]] = []
        self._current_rows: list[list[str]] | None = None
        self._in_heading = False
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"h2", "h3"}:
            self._heading = []
            self._in_heading = True
        elif tag == "table":
            self._current_rows = []
        elif tag in {"th", "td"}:
            self._cell = []
            self._in_cell = True
        elif tag == "tr" and self._current_rows is not None:
            self._row = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h2", "h3"}:
            self.section = "".join(self._heading).strip()
            self._in_heading = False
        elif tag in {"th", "td"}:
            self._row.append("".join(self._cell).strip())
            self._cell = []
            self._in_cell = False
        elif tag == "tr" and self._current_rows is not None and self._row:
            self._current_rows.append(self._row)
            self._row = []
        elif tag == "table" and self._current_rows is not None:
            self._table.append((self.section, self._current_rows))
            self._current_rows = None

    def handle_data(self, data: str) -> None:
        if self._in_heading:
            self._heading.append(data)
        elif self._in_cell:
            self._cell.append(data)

    @property
    def tables(self) -> list[tuple[str, list[list[str]]]]:
        return self._table


def _find_table(
    tables: list[tuple[str, list[list[str]]]], section_terms: tuple[str, ...]
) -> list[list[str]]:
    for section, rows in tables:
        if any(term in section for term in section_terms):
            return rows
    msg = f"Documentation table not found for sections: {section_terms}"
    raise RuntimeError(msg)


def _header_indexes(header: list[str]) -> tuple[int, int]:
    normalized = [cell.replace(" ", "") for cell in header]
    try:
        name_index = normalized.index("模型ID")
    except ValueError as error:
        raise RuntimeError("Documentation table has no model ID column") from error
    try:
        label_index = normalized.index("模型")
    except ValueError as error:
        raise RuntimeError("Documentation table has no model name column") from error
    return label_index, name_index


def _endpoint_models(rows: list[list[str]]) -> list[ModelRecord]:
    if not rows:
        raise RuntimeError("Documentation model table is empty")
    label_index, name_index = _header_indexes(rows[0])
    models: list[ModelRecord] = []
    for row in rows[1:]:
        if len(row) > max(label_index, name_index) and row[name_index]:
            models.append(ModelRecord(row[name_index], "", row[label_index]))
    return models


def _free_names(rows: list[list[str]]) -> set[str]:
    if not rows:
        raise RuntimeError("Documentation pricing table is empty")
    header = [cell.replace(" ", "") for cell in rows[0]]
    try:
        label_index = header.index("模型")
        input_index = header.index("输入")
        output_index = header.index("输出")
    except ValueError as error:
        raise RuntimeError("Documentation pricing table has unexpected columns") from error
    return {
        row[label_index]
        for row in rows[1:]
        if len(row) > max(label_index, input_index, output_index)
        and row[input_index].lower() == "free"
        and row[output_index].lower() == "free"
    }


def extract_models(html: str, provider: str) -> list[ModelRecord]:
    """Extract provider model IDs from one OpenCode documentation page."""
    parser = _TableParser()
    parser.feed(html)
    endpoint_rows = _find_table(parser.tables, ("API 端点", "端点"))
    models = _endpoint_models(endpoint_rows)
    if provider == "go":
        return [ModelRecord(model.name, "go", model.label) for model in models]
    if provider == "zen":
        pricing_rows = _find_table(parser.tables, ("定价",))
        free_names = _free_names(pricing_rows)
        return [
            ModelRecord(model.name, "zen")
            for model in models
            if model.label in free_names
        ]
    raise ValueError(f"Unsupported provider: {provider}")


def extract_minimax_models(html: str) -> list[ModelRecord]:
    """Extract MiniMax language model IDs from the API overview page.

    Only tables under the "语言模型" heading are considered; the H3 video model
    lives under its own heading and is naturally excluded.
    """
    parser = _TableParser()
    parser.feed(html)
    rows = _find_table(parser.tables, ("语言模型",))
    if not rows:
        raise RuntimeError("MiniMax language model table not found")
    header = [cell.replace(" ", "") for cell in rows[0]]
    try:
        name_index = header.index("模型名称")
    except ValueError as error:
        raise RuntimeError("MiniMax table has no 模型名称 column") from error
    models: list[ModelRecord] = []
    for row in rows[1:]:
        if len(row) > name_index and row[name_index]:
            models.append(ModelRecord(row[name_index], "minimax"))
    return models


def _fetch(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(request, timeout=30, context=context) as response:
            return response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"Failed to fetch {url}: {error}") from error


def _write_config(
    output: Path,
    go_models: list[ModelRecord],
    zen_models: list[ModelRecord],
    minimax_models: list[ModelRecord],
) -> None:
    config = {
        "auth_file": "~/.local/share/opencode/auth.json",
        "endpoints": {
            "zen": {"base_url": "https://opencode.ai/zen/v1", "auth_provider": "opencode-go"},
            "go": {"base_url": "https://opencode.ai/zen/go/v1", "auth_provider": "opencode-go"},
            "minimax": {"base_url": "https://api.minimaxi.com/v1", "auth_provider": "minimax-cn-coding-plan"},
        },
        "models": [
            {"name": model.name, "endpoint": model.endpoint}
            for model in [*go_models, *zen_models, *minimax_models]
        ],
        "extractor": {
            "max_retries": 2,
            "content_retries": 2,
            "timeout": 120,
            "temperature": 0.2,
            "max_tokens_per_chunk": 2000,
            "max_total_prompt_tokens": 30000,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> int:  # noqa: BROAD_EXCEPT_OK
    parser = argparse.ArgumentParser(description="Update config/opencode_models.yaml from OpenCode/MiniMax docs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--go-url", default=GO_DOC_URL)
    parser.add_argument("--zen-url", default=ZEN_DOC_URL)
    parser.add_argument("--minimax-url", default=MINIMAX_DOC_URL)
    args = parser.parse_args()
    try:
        go_models = extract_models(_fetch(args.go_url), "go")
        zen_models = extract_models(_fetch(args.zen_url), "zen")
        minimax_models = extract_minimax_models(_fetch(args.minimax_url))
        _write_config(args.output, go_models, zen_models, minimax_models)
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {len(go_models)} Go models, {len(zen_models)} free Zen models, {len(minimax_models)} MiniMax models to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
