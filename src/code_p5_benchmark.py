#!/usr/bin/env python3
"""
Phase 5 模型对比测试: 三元组抽取成功率

对比 deepseek-v4-flash-free / nemotron-3-ultra-free / mimo-v2.5-free
三个模型在同一批 task 上的三元组抽取成功率、解析质量和耗时。

用法:
  export OPENCODE_ZEN_API_KEY=$(python3 -c "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")
  python code_p5_benchmark.py
  python code_p5_benchmark.py --limit 5
  python code_p5_benchmark.py --models deepseek-v4-flash-free,nemotron-3-ultra-free
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from loguru import logger

# 复用已有 prompt 和解析工具
from code_p5_kg_builder import (
    _extract_json_from_response,
    _repair_truncated_json,
)
from code_update_prompt_utils import load_prompt
from code_p2_models import Task


# ============================================================
# 配置
# ============================================================

MODELS = [
    "deepseek-v4-flash-free",
    "nemotron-3-ultra-free",
    "mimo-v2.5-free",
]

BASE_URL = "https://opencode.ai/zen/v1"
MAX_TOKENS = 8000      # reasoning 模型需要足够大的 max_tokens
MAX_RETRIES = 2
TIMEOUT = 120


# ============================================================
# LLM 调用 & 解析
# ============================================================

def call_llm(client: OpenAI, model: str, prompt: str) -> dict:
    """
    调用 LLM, 返回详细诊断信息

    Returns:
        {
            "content": str,
            "has_reasoning": bool,
            "reasoning_preview": str,  # 前 200 字符
            "content_empty": bool,
            "elapsed": float,
        }
    """
    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=MAX_TOKENS,
    )
    elapsed = time.time() - t0

    msg = response.choices[0].message
    content = msg.content or ""

    # 检查 reasoning_content
    reasoning = ""
    if hasattr(msg, 'model_extra') and msg.model_extra:
        reasoning = msg.model_extra.get('reasoning_content', '') or ""

    content_empty = not content.strip()

    # 如果 content 为空, 尝试从 reasoning_content 提取 JSON
    if content_empty and reasoning.strip():
        logger.info(f"  [{model}] content 为空, 尝试从 reasoning_content 提取 JSON")
        extracted = _extract_json_from_response(reasoning)
        try:
            json.loads(extracted)
            content = extracted
            logger.info(f"  [{model}] 从 reasoning_content 成功提取 JSON")
        except json.JSONDecodeError:
            logger.warning(f"  [{model}] reasoning_content 中未找到有效 JSON")
            content = reasoning  # fallback: 使用原始 reasoning

    return {
        "content": content.strip(),
        "has_reasoning": bool(reasoning.strip()),
        "reasoning_preview": reasoning[:200] if reasoning else "",
        "content_empty": content_empty,
        "elapsed": elapsed,
    }


def parse_triples(llm_output: str) -> list[dict]:
    """从 LLM 输出中解析三元组"""
    json_str = _extract_json_from_response(llm_output)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        repaired = _repair_truncated_json(json_str)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            return []

    triples_data = data.get("triples", [])
    if not isinstance(triples_data, list):
        return []

    valid = []
    for t in triples_data:
        head = str(t.get("head", "")).strip()
        relation = str(t.get("relation", "")).strip()
        tail = str(t.get("tail", "")).strip()
        if head and relation and tail:
            valid.append({
                "head": head,
                "relation": relation,
                "tail": tail,
                "confidence": float(t.get("confidence", 0.5)),
            })
    return valid


def benchmark_model(
    client: OpenAI,
    model: str,
    tasks: list[Task],
    concurrency: int = 4,
) -> dict:
    """测试单个模型在三元组抽取上的表现"""
    logger.info(f"\n{'='*60}")
    logger.info(f"测试模型: {model}")
    logger.info(f"{'='*60}")

    results = {
        "model": model,
        "total": len(tasks),
        "success": 0,
        "failed": 0,
        "total_triples": 0,
        "content_empty_count": 0,
        "reasoning_detected_count": 0,
        "json_from_reasoning_count": 0,
        "per_task": [],
        "total_time": 0,
    }

    t_total = time.time()

    def _process(task: Task) -> dict:
        prompt = load_prompt("TRIPLE_EXTRACTION_PROMPT").format(task_summary=task.task_summary)
        try:
            resp = call_llm(client, model, prompt)
            triples = parse_triples(resp["content"])
            return {
                "task_id": task.task_id,
                "task_label": task.task_label,
                "success": len(triples) > 0,
                "n_triples": len(triples),
                "triples": triples,
                "content_empty": resp["content_empty"],
                "has_reasoning": resp["has_reasoning"],
                "elapsed": resp["elapsed"],
                "error": None,
            }
        except Exception as e:
            return {
                "task_id": task.task_id,
                "task_label": task.task_label,
                "success": False,
                "n_triples": 0,
                "triples": [],
                "content_empty": True,
                "has_reasoning": False,
                "elapsed": 0,
                "error": str(e),
            }

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_process, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results["per_task"].append(r)

            if r["success"]:
                results["success"] += 1
                results["total_triples"] += r["n_triples"]
            else:
                results["failed"] += 1

            if r["content_empty"]:
                results["content_empty_count"] += 1
            if r["has_reasoning"]:
                results["reasoning_detected_count"] += 1
            if r["content_empty"] and r["has_reasoning"]:
                results["json_from_reasoning_count"] += 1

            status = "OK" if r["success"] else "FAIL"
            err_info = f" ({r['error'][:60]})" if r.get("error") else ""
            logger.info(
                f"  [{i}/{len(tasks)}] {r['task_label']}: "
                f"{status}, {r['n_triples']} triples, "
                f"{r['elapsed']:.1f}s{err_info}"
            )

    results["total_time"] = time.time() - t_total

    # 汇总
    rate = results["success"] / results["total"] * 100 if results["total"] else 0
    logger.info(f"\n--- {model} 结果 ---")
    logger.info(f"  成功率: {rate:.0f}% ({results['success']}/{results['total']})")
    logger.info(f"  三元组总数: {results['total_triples']}")
    logger.info(f"  content 为空: {results['content_empty_count']} 次")
    logger.info(f"  检测到 reasoning_content: {results['reasoning_detected_count']} 次")
    logger.info(f"  从 reasoning 提取 JSON: {results['json_from_reasoning_count']} 次")
    logger.info(f"  总耗时: {results['total_time']:.1f}s")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 5 模型对比测试")
    parser.add_argument("--limit", type=int, default=10,
                        help="测试 task 数量 (默认 10)")
    parser.add_argument("--models", type=str, default=None,
                        help="模型列表, 逗号分隔")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="并发数 (默认 4)")
    args = parser.parse_args()

    # API Key
    api_key = os.environ.get("OPENCODE_ZEN_API_KEY", "")
    if not api_key:
        logger.error("OPENCODE_ZEN_API_KEY 未设置")
        sys.exit(1)

    # 加载 tasks
    tasks_file = "./output/tasks.jsonl"
    logger.info(f"加载 tasks: {tasks_file}")
    tasks = []
    with open(tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(Task.from_dict(json.loads(line)))

    if args.limit:
        tasks = tasks[:args.limit]
    logger.info(f"测试 {len(tasks)} 个 task, 并发={args.concurrency}")

    # 模型列表
    models = args.models.split(",") if args.models else MODELS

    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=TIMEOUT)

    # 逐模型测试
    all_results = []
    for model in models:
        result = benchmark_model(client, model, tasks, concurrency=args.concurrency)
        all_results.append(result)

    # ============================================================
    # 对比表
    # ============================================================

    print("\n" + "=" * 90)
    print("模型对比结果")
    print("=" * 90)
    header = f"{'模型':<30} {'成功率':>10} {'三元组':>8} {'content空':>10} {'reasoning':>10} {'耗时':>8}"
    print(header)
    print("-" * 90)

    for r in all_results:
        rate = r["success"] / r["total"] * 100 if r["total"] else 0
        print(
            f"{r['model']:<30} "
            f"{rate:>7.0f}% "
            f"{r['total_triples']:>8d} "
            f"{r['content_empty_count']:>10d} "
            f"{r['reasoning_detected_count']:>10d} "
            f"{r['total_time']:>7.1f}s"
        )

    # 保存详细结果
    output_file = "./output/benchmark_results.json"
    Path("./output").mkdir(exist_ok=True)
    # 清理不可序列化的内容 (triples 保留)
    serializable = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k != "per_task"}
        sr["per_task_summary"] = [
            {
                "task_id": t["task_id"],
                "task_label": t["task_label"],
                "success": t["success"],
                "n_triples": t["n_triples"],
                "content_empty": t["content_empty"],
                "has_reasoning": t["has_reasoning"],
                "elapsed": t["elapsed"],
                "error": t.get("error"),
            }
            for t in r["per_task"]
        ]
        # 保存每个模型的成功三元组示例
        sr["sample_triples"] = []
        for t in r["per_task"]:
            if t["success"]:
                sr["sample_triples"].extend(t["triples"][:3])
        serializable.append(sr)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {output_file}")


if __name__ == "__main__":
    main()
