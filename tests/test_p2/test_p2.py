"""
Phase 2 多模型对比测试
测试 LLM 模型在任务提取 (CoT 三步法) 上的效果

评估维度:
  - 耗时 (latency): 每 session 平均耗时 + 总耗时
  - 成功率: JSON 解析成功 + content 非空
  - 完整性: chunk_summary 覆盖率 + chunk 任务归属覆盖率
  - 质量: task label 平均长度, task summary 平均长度, task 数量合理性

模型 / 端点 / 认证配置集中在 tests/test_p2/config.yaml 中,
按 models 列表顺序依次加载测试。

用法:
  # 直接运行 (读取 tests/test_p2/config.yaml, 全部模型, 5 个 session)
  python3 tests/test_p2/test_p2.py

  # 只测指定模型 (覆盖 config.yaml 的 models 列表, 端点/认证仍从 config 解析)
  python3 tests/test_p2/test_p2.py --models hy3 nemotron-3-ultra-free

  # 全量 session + 8 并发
  python3 tests/test_p2/test_p2.py --sessions 100 --concurrency 8

  # 指定配置文件
  python3 tests/test_p2/test_p2.py --config tests/test_p2/config.yaml

  # 指定输出根目录 (默认 tests/test_p2/, 实际结果写入带日期的子文件夹)
  python3 tests/test_p2/test_p2.py --output-dir tests/test_p2
"""

import argparse
import copy
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

# 添加 src 到路径 (tests/test_p2/ -> tests/ -> project root)
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "src"))

from loguru import logger

from code_p1_models import Chunk, CleanedToolCall
from code_p2_task_extractor import TaskExtractor
from code_p2_models import Task


# ============================================================
# 配置
# ============================================================

DEFAULT_CONFIG_FILE = str(Path(__file__).resolve().parent / "config.yaml")
CHUNKS_FILE = str(_project_root / "output" / "chunks.jsonl")


def load_test_config(config_path: str) -> dict:
    """加载测试配置 (模型 / 端点 / 认证)"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_api_key(auth_file: str, provider: str) -> str:
    """从 opencode auth.json 中解析指定 provider 的 API Key"""
    path = Path(os.path.expanduser(auth_file))
    if not path.exists():
        logger.error(f"认证文件不存在: {path}")
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(provider, {}).get("key", "")
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"解析认证文件失败 ({provider}): {e}")
        return ""


def build_model_specs(cfg: dict, override_models: list[str] | None) -> list[dict]:
    """
    根据配置构建待测模型规格列表, 每项含:
      {name, base_url, api_key, auth_provider, endpoint}
    按 config.yaml models 顺序; 若 --models 覆盖, 则用覆盖列表的顺序,
    端点/认证从 config.endpoints 中按模型的 endpoint 字段解析
    (覆盖模型若不在 config 中, 默认走第一个端点)。
    """
    endpoints = cfg.get("endpoints", {})
    auth_file = cfg.get("auth_file", "~/.local/share/opencode/auth.json")
    configured = {m["name"]: m for m in cfg.get("models", [])}

    if override_models:
        default_ep = next(iter(endpoints), None)
        model_entries = []
        for name in override_models:
            if name in configured:
                model_entries.append(configured[name])
            else:
                logger.warning(
                    f"模型 {name} 不在 config.yaml 中, 默认使用端点 '{default_ep}'"
                )
                model_entries.append({"name": name, "endpoint": default_ep})
    else:
        model_entries = cfg.get("models", [])

    # 缓存每个 provider 的 key, 避免重复读盘
    key_cache: dict[str, str] = {}
    specs = []
    for entry in model_entries:
        name = entry["name"]
        ep_name = entry.get("endpoint")
        ep = endpoints.get(ep_name, {})
        base_url = ep.get("base_url", "")
        provider = ep.get("auth_provider", "")
        if provider not in key_cache:
            key_cache[provider] = resolve_api_key(auth_file, provider)
        specs.append({
            "name": name,
            "endpoint": ep_name,
            "base_url": base_url,
            "auth_provider": provider,
            "api_key": key_cache[provider],
        })
    return specs


# ============================================================
# 辅助函数
# ============================================================

def load_chunks(path: str) -> list[Chunk]:
    """从 JSONL 加载 chunks"""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            chunk = Chunk(
                chunk_id=d["chunk_id"],
                session_id=d["session_id"],
                turn_index=d["turn_index"],
                user_message=d["user_message"],
                assistant_messages=d.get("assistant_messages", []),
                tool_calls=[CleanedToolCall(**tc) for tc in d.get("tool_calls", [])],
                mcp_calls=[CleanedToolCall(**mc) for mc in d.get("mcp_calls", [])],
                raw_size_tokens=d.get("raw_size_tokens", 0),
                cleaned_size_tokens=d.get("cleaned_size_tokens", 0),
                created_at=d.get("created_at"),
                task_summary=None,  # 清空, 测试时重新生成
            )
            chunks.append(chunk)
    return chunks


def select_sessions(
    chunks: list[Chunk], num_sessions: int = 5
) -> dict[str, list[Chunk]]:
    """选择有代表性的 session (不同大小)"""
    groups = defaultdict(list)
    for c in chunks:
        groups[c.session_id].append(c)
    for sid in groups:
        groups[sid].sort(key=lambda x: x.turn_index)

    # 按 chunk 数排序, 均匀采样
    sorted_sessions = sorted(groups.items(), key=lambda x: len(x[1]))
    n = len(sorted_sessions)
    if num_sessions >= n:
        return dict(sorted_sessions)

    # 等间距采样
    step = n / num_sessions
    selected = []
    for i in range(num_sessions):
        idx = int(i * step)
        selected.append(sorted_sessions[idx])

    return dict(selected)


def deep_copy_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """深拷贝 chunks (避免 task_summary 被跨模型污染)"""
    return [
        Chunk(
            chunk_id=c.chunk_id,
            session_id=c.session_id,
            turn_index=c.turn_index,
            user_message=c.user_message,
            assistant_messages=list(c.assistant_messages),
            tool_calls=[
                CleanedToolCall(
                    type=tc.type, summary=tc.summary, ok=tc.ok, error=tc.error
                )
                for tc in c.tool_calls
            ],
            mcp_calls=[
                CleanedToolCall(
                    type=mc.type, summary=mc.summary, ok=mc.ok, error=mc.error
                )
                for mc in c.mcp_calls
            ],
            raw_size_tokens=c.raw_size_tokens,
            cleaned_size_tokens=c.cleaned_size_tokens,
            created_at=c.created_at,
            task_summary=None,
        )
        for c in chunks
    ]


def evaluate_quality(tasks: list[Task], chunks: list[Chunk]) -> dict:
    """评估提取质量"""
    if not tasks:
        return {
            "task_count": 0,
            "avg_label_len": 0,
            "avg_summary_len": 0,
            "chunk_coverage": 0.0,
            "duplicate_chunks": 0,
        }

    # Task label 和 summary 长度
    label_lens = [len(t.task_label) for t in tasks if t.task_label]
    summary_lens = [len(t.task_summary) for t in tasks if t.task_summary]

    # Chunk 覆盖率
    all_chunk_ids = {c.chunk_id for c in chunks}
    assigned = set()
    chunk_count = defaultdict(int)
    for t in tasks:
        for cid in t.chunk_ids:
            assigned.add(cid)
            chunk_count[cid] += 1

    coverage = len(assigned & all_chunk_ids) / len(all_chunk_ids) if all_chunk_ids else 0
    duplicates = sum(1 for v in chunk_count.values() if v > 1)

    return {
        "task_count": len(tasks),
        "avg_label_len": sum(label_lens) / len(label_lens) if label_lens else 0,
        "avg_summary_len": sum(summary_lens) / len(summary_lens) if summary_lens else 0,
        "chunk_coverage": coverage,
        "duplicate_chunks": duplicates,
    }


# ============================================================
# 主测试流程
# ============================================================

def run_test(
    num_sessions: int = 5,
    output_dir: str = "tests/test_p2",
    concurrency: int = 1,
    config_path: str = DEFAULT_CONFIG_FILE,
    override_models: list[str] | None = None,
):
    """运行多模型对比测试"""

    # 0. 加载测试配置 (模型 / 端点 / 认证)
    logger.info(f"加载测试配置: {config_path}")
    cfg = load_test_config(config_path)
    model_specs = build_model_specs(cfg, override_models)
    ext_cfg = cfg.get("extractor", {})

    if not model_specs:
        logger.error("没有可测试的模型, 请检查 config.yaml 的 models 列表")
        sys.exit(1)

    # 1. 校验每个模型的 API key
    for spec in model_specs:
        if not spec["api_key"]:
            logger.error(
                f"模型 {spec['name']} (provider={spec['auth_provider']}) "
                f"未能从认证文件解析到 API Key, 请检查 "
                f"{cfg.get('auth_file')}"
            )
            sys.exit(1)
    logger.info(
        "待测模型: "
        + ", ".join(f"{s['name']}@{s['endpoint']}" for s in model_specs)
    )

    # 2. 加载数据
    logger.info(f"加载 chunks: {CHUNKS_FILE}")
    all_chunks = load_chunks(CHUNKS_FILE)
    logger.info(f"加载 {len(all_chunks)} 个 chunk")

    # 3. 选择样本 sessions
    sessions_chunks = select_sessions(all_chunks, num_sessions)
    logger.info(f"选择 {len(sessions_chunks)} 个 session 进行测试:")
    for sid, chunks in sessions_chunks.items():
        logger.info(f"  {sid}: {len(chunks)} chunks")

    # 4. 逐模型测试
    results = {}  # model -> {session_id -> result}
    model_stats = {}  # model -> aggregate stats
    models = [s["name"] for s in model_specs]

    for spec in model_specs:
        model = spec["name"]
        logger.info("=" * 60)
        logger.info(f"测试模型: {model}  (端点={spec['endpoint']}: {spec['base_url']})")
        logger.info("=" * 60)

        model_results = {}
        total_tasks = 0
        total_latency = 0.0
        n_success = 0
        n_incomplete = 0
        n_failed = 0
        all_quality = []

        extractor = TaskExtractor(
            model=model,
            api_key=spec["api_key"],
            base_url=spec["base_url"],
            max_retries=ext_cfg.get("max_retries", 2),
            content_retries=ext_cfg.get("content_retries", 2),
            timeout=ext_cfg.get("timeout", 120),
            max_tokens_per_chunk=ext_cfg.get("max_tokens_per_chunk", 2000),
            max_total_prompt_tokens=ext_cfg.get("max_total_prompt_tokens", 30000),
            concurrency=concurrency,
            temperature=ext_cfg.get("temperature", 0.3),
        )

        for sid, chunks in sessions_chunks.items():
            # 深拷贝, 防止 task_summary 跨模型污染
            test_chunks = deep_copy_chunks(chunks)

            logger.info(f"  Session {sid[:20]}... ({len(test_chunks)} chunks)")
            t0 = time.time()
            try:
                result = extractor.extract_tasks(sid, test_chunks)
                latency = time.time() - t0
                result["latency_s"] = round(latency, 2)

                # 质量评估
                quality = evaluate_quality(result["tasks"], test_chunks)
                result["quality"] = quality

                # chunk_summary 覆盖率
                summary_coverage = (
                    result["summaries_written"] / result["total_chunks"]
                    if result["total_chunks"] > 0
                    else 0
                )
                result["summary_coverage"] = round(summary_coverage, 3)

                model_results[sid] = result
                total_tasks += len(result["tasks"])
                total_latency += latency

                if result["status"] == "success":
                    n_success += 1
                elif result["status"] == "incomplete":
                    n_incomplete += 1
                else:
                    n_failed += 1

                all_quality.append(quality)

                logger.info(
                    f"    -> {result['status']}, "
                    f"{len(result['tasks'])} tasks, "
                    f"summary={result['summaries_written']}/{result['total_chunks']}, "
                    f"{latency:.1f}s"
                )
            except Exception as e:
                latency = time.time() - t0
                logger.error(f"    -> ERROR: {e}")
                model_results[sid] = {
                    "status": "error",
                    "error": str(e),
                    "latency_s": round(latency, 2),
                    "tasks": [],
                    "summaries_written": 0,
                    "total_chunks": len(test_chunks),
                    "quality": evaluate_quality([], test_chunks),
                    "summary_coverage": 0,
                }
                total_latency += latency
                n_failed += 1
                all_quality.append(evaluate_quality([], test_chunks))

        # 汇总统计
        n_sessions = len(sessions_chunks)
        avg_latency = total_latency / n_sessions if n_sessions else 0

        # 平均质量指标
        avg_task_count = sum(q["task_count"] for q in all_quality) / len(all_quality) if all_quality else 0
        avg_label_len = sum(q["avg_label_len"] for q in all_quality) / len(all_quality) if all_quality else 0
        avg_summary_len = sum(q["avg_summary_len"] for q in all_quality) / len(all_quality) if all_quality else 0
        avg_chunk_cov = sum(q["chunk_coverage"] for q in all_quality) / len(all_quality) if all_quality else 0

        # summary coverage 从 result 取
        summary_cov_vals = [
            r.get("summary_coverage", 0)
            for r in model_results.values()
            if r.get("summary_coverage") is not None
        ]
        avg_summary_cov = sum(summary_cov_vals) / len(summary_cov_vals) if summary_cov_vals else 0

        model_stats[model] = {
            "total_sessions": n_sessions,
            "success": n_success,
            "incomplete": n_incomplete,
            "failed": n_failed,
            "success_rate": round(n_success / n_sessions, 3) if n_sessions else 0,
            "total_latency_s": round(total_latency, 2),
            "avg_latency_s": round(avg_latency, 2),
            "total_tasks": total_tasks,
            "avg_tasks_per_session": round(avg_task_count, 2),
            "avg_label_len": round(avg_label_len, 1),
            "avg_summary_len": round(avg_summary_len, 1),
            "avg_chunk_coverage": round(avg_chunk_cov, 3),
            "avg_summary_coverage": round(avg_summary_cov, 3),
        }

        results[model] = model_results

    # 5. 输出对比表
    print_comparison_table(model_stats)

    # 6. 保存详细结果到带日期的子文件夹
    #    tests/test_p2/run_<YYYY-MM-DD_HH-MM>/
    run_ts = datetime.now()
    run_dir = Path(output_dir) / f"run_{run_ts.strftime('%Y-%m-%d_%H-%M')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "test_p2_results.json"

    # 序列化 results (Task 对象转 dict)
    serializable_results = {}
    for model, sessions in results.items():
        serializable_results[model] = {}
        for sid, result in sessions.items():
            r = dict(result)
            r["tasks"] = [t.to_dict() if hasattr(t, "to_dict") else t for t in r.get("tasks", [])]
            serializable_results[model][sid] = r

    output_data = {
        "test_time": run_ts.isoformat(),
        "config_file": str(config_path),
        "models": models,
        "model_specs": [
            {"name": s["name"], "endpoint": s["endpoint"], "base_url": s["base_url"]}
            for s in model_specs
        ],
        "num_sessions": num_sessions,
        "concurrency": concurrency,
        "sessions": {sid: len(chunks) for sid, chunks in sessions_chunks.items()},
        "model_stats": model_stats,
        "detailed_results": serializable_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"\n详细结果已保存到: {output_path}")

    # 7. 生成 Markdown 报告 (与 test_p2_models.md 同格式/位置)
    report_path = run_dir / "test_p2_models.md"
    write_markdown_report(report_path, output_data, model_specs, config_path)
    logger.info(f"测试报告已生成: {report_path}")

    return model_stats


def write_markdown_report(
    report_path: Path,
    output_data: dict,
    model_specs: list[dict],
    config_path: str,
):
    """生成 Markdown 对比报告 (结构对齐 test_p2_models.md)"""
    stats = output_data["model_stats"]
    models = output_data["models"]
    sessions = output_data["sessions"]
    run_time = output_data["test_time"]

    lines: list[str] = []
    lines.append("# Phase 2 多模型对比测试报告\n")
    lines.append("## 测试概要\n")
    lines.append(f"- **测试时间**: {run_time}")
    lines.append(f"- **配置文件**: `{config_path}`")
    lines.append(f"- **测试样本**: {output_data['num_sessions']} 个 session（实测 {len(sessions)} 个）")
    lines.append(f"- **并发数**: {output_data['concurrency']}")
    lines.append("- **测试模型 / 端点**:")
    for s in model_specs:
        lines.append(f"  - `{s['name']}` @ {s['endpoint']} (`{s['base_url']}`)")
    lines.append("- **测试脚本**: `tests/test_p2/test_p2.py`\n")

    lines.append("## 复现方法\n")
    lines.append("```bash")
    lines.append("cd ~/Desktop/oc_sess_graph")
    lines.append("# 模型 / 端点 / 认证配置见 tests/test_p2/config.yaml")
    lines.append(
        f"python3 tests/test_p2/test_p2.py --sessions {output_data['num_sessions']} "
        f"--concurrency {output_data['concurrency']}"
    )
    lines.append("```\n")

    # 汇总对比表
    lines.append("## 汇总对比表\n")
    header = "| 指标 | " + " | ".join(models) + " |"
    sep = "|------|" + "|".join(["------"] * len(models)) + "|"
    lines.append(header)
    lines.append(sep)

    def row(label: str, fmt):
        cells = " | ".join(fmt(stats[m]) for m in models)
        return f"| **{label}** | {cells} |"

    lines.append(row("成功率", lambda s: f"{s['success_rate']:.0%} ({s['success']}/{s['total_sessions']})"))
    lines.append(row("平均耗时(s)", lambda s: f"{s['avg_latency_s']:.1f}"))
    lines.append(row("总耗时(s)", lambda s: f"{s['total_latency_s']:.1f}"))
    lines.append(row("总 Task 数", lambda s: f"{s['total_tasks']}"))
    lines.append(row("平均 Task/session", lambda s: f"{s['avg_tasks_per_session']:.1f}"))
    lines.append(row("Label 均长(字)", lambda s: f"{s['avg_label_len']:.0f}"))
    lines.append(row("Summary 均长(字)", lambda s: f"{s['avg_summary_len']:.0f}"))
    lines.append(row("Chunk 覆盖率", lambda s: f"{s['avg_chunk_coverage']:.0%}"))
    lines.append(row("Summary 覆盖率", lambda s: f"{s['avg_summary_coverage']:.0%}"))
    lines.append(row("不完整", lambda s: f"{s['incomplete']}"))
    lines.append(row("失败", lambda s: f"{s['failed']}"))
    lines.append("")

    # 分析结论
    lines.append("## 分析结论\n")
    best_speed = min(stats.items(), key=lambda x: x[1]["avg_latency_s"])
    best_quality = max(stats.items(), key=lambda x: x[1]["avg_summary_len"])
    best_cov = max(stats.items(), key=lambda x: x[1]["avg_summary_coverage"])
    best_success = max(stats.items(), key=lambda x: x[1]["success_rate"])
    lines.append(f"- **最快**: `{best_speed[0]}` (平均 {best_speed[1]['avg_latency_s']:.1f}s/session)")
    lines.append(f"- **Summary 最详细**: `{best_quality[0]}` (平均 {best_quality[1]['avg_summary_len']:.0f} 字)")
    lines.append(f"- **Summary 覆盖最高**: `{best_cov[0]}` ({best_cov[1]['avg_summary_coverage']:.0%})")
    lines.append(f"- **成功率最高**: `{best_success[0]}` ({best_success[1]['success_rate']:.0%})\n")

    lines.append("## 附录\n")
    lines.append("- 详细 JSON 结果: `test_p2_results.json` (同目录)")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def print_comparison_table(model_stats: dict):
    """打印对比表"""
    print("\n" + "=" * 80)
    print("Phase 2 多模型对比测试结果")
    print("=" * 80)

    # 表头
    headers = [
        "模型",
        "成功率",
        "平均耗时(s)",
        "总耗时(s)",
        "平均Tasks",
        "Label长度",
        "Summary长度",
        "Chunk覆盖",
        "Summary覆盖",
    ]
    col_widths = [26, 8, 12, 12, 10, 10, 12, 10, 12]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    for model, stats in model_stats.items():
        row = [
            model[:26],
            f"{stats['success_rate']:.0%}",
            f"{stats['avg_latency_s']:.1f}",
            f"{stats['total_latency_s']:.1f}",
            f"{stats['avg_tasks_per_session']:.1f}",
            f"{stats['avg_label_len']:.0f}",
            f"{stats['avg_summary_len']:.0f}",
            f"{stats['avg_chunk_coverage']:.0%}",
            f"{stats['avg_summary_coverage']:.0%}",
        ]
        print(" | ".join(v.ljust(w) for v, w in zip(row, col_widths)))

    print("=" * 80)

    # 分析
    print("\n分析:")
    best_speed = min(model_stats.items(), key=lambda x: x[1]["avg_latency_s"])
    best_quality = max(model_stats.items(), key=lambda x: x[1]["avg_summary_len"])
    best_coverage = max(model_stats.items(), key=lambda x: x[1]["avg_summary_coverage"])
    best_success = max(model_stats.items(), key=lambda x: x[1]["success_rate"])

    print(f"  最快: {best_speed[0]} (平均 {best_speed[1]['avg_latency_s']:.1f}s/session)")
    print(f"  最详细 Summary: {best_quality[0]} (平均 {best_quality[1]['avg_summary_len']:.0f} 字)")
    print(f"  最高 Summary 覆盖: {best_coverage[0]} ({best_coverage[1]['avg_summary_coverage']:.0%})")
    print(f"  最高成功率: {best_success[0]} ({best_success[1]['success_rate']:.0%})")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 多模型对比测试")
    parser.add_argument(
        "--sessions", type=int, default=5,
        help="测试的 session 数量 (默认 5, 从不同大小中采样)",
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_FILE,
        help="测试配置文件路径 (模型/端点/认证), 默认 tests/test_p2/config.yaml",
    )
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).resolve().parent),
        help="结果输出根目录 (实际写入其下带日期的 run_ 子文件夹)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="每个模型的并发线程数 (默认 1=串行)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="覆盖 config.yaml 的模型列表 (端点/认证仍从 config 解析)",
    )
    args = parser.parse_args()

    # 日志配置
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    run_test(
        num_sessions=args.sessions,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        config_path=args.config,
        override_models=args.models,
    )
