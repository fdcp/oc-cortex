"""用途：验证 OpenCode 模型池文档解析和免费模型筛选逻辑。

使用方式：运行 `python3 tests/test_opencode_models.py`。

输入：测试内置的最小 HTML 表格样例，不访问网络，也不读写生成配置。
期望输出：命令无输出并返回成功状态，表示 Go 全量提取和 Zen 免费筛选均通过。
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from code_update_opencode_models import extract_models


def test_extract_models_keeps_all_go_models() -> None:
    html = """
    <h2>API 端点</h2>
    <table>
      <tr><th>模型</th><th>模型 ID</th><th>端点</th></tr>
      <tr><td>GLM 5</td><td>glm-5</td><td>chat</td></tr>
      <tr><td>Hy3</td><td>hy3</td><td>chat</td></tr>
    </table>
    """

    result = extract_models(html, provider="go")

    assert [model.name for model in result] == ["glm-5", "hy3"]


def test_extract_models_keeps_only_free_zen_models() -> None:
    html = """
    <h2>端点</h2>
    <table>
      <tr><th>模型</th><th>模型 ID</th><th>端点</th></tr>
      <tr><td>Big Pickle</td><td>big-pickle</td><td>chat</td></tr>
      <tr><td>GPT 5.5</td><td>gpt-5.5</td><td>responses</td></tr>
    </table>
    <h2>定价</h2>
    <table>
      <tr><th>模型</th><th>输入</th><th>输出</th></tr>
      <tr><td>Big Pickle</td><td>Free</td><td>Free</td></tr>
      <tr><td>GPT 5.5</td><td>$5.00</td><td>$30.00</td></tr>
    </table>
    """

    result = extract_models(html, provider="zen")

    assert [model.name for model in result] == ["big-pickle"]
