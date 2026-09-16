#!/usr/bin/env bash
# Graph-RAG trace 一键运行脚本
# 自动设置环境变量并运行两个 trace 工具

set -e

cd "$(dirname "$0")/../.."

if [ -f "$HOME/.local/share/opencode/auth.json" ]; then
    export OPENCODE_ZEN_API_KEY=$(python3 -c \
        "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")
    echo "[setup] OPENCODE_ZEN_API_KEY 已从 auth.json 加载 (len=${#OPENCODE_ZEN_API_KEY})"
else
    echo "[setup] WARN: $HOME/.local/share/opencode/auth.json 不存在"
    echo "[setup] 图谱通道会因 API key 失败退化为纯向量"
fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 默认 query 列表
DEFAULT_QUERIES=(
    "分布式训练"
    "FlashAttention"
    "序列并行"
    "BF16 混合精度"
)

if [ "$1" == "--compare-batch" ] || [ "$1" == "-b" ]; then
    shift
    QUERY_FILE="${1:-tests/graph_rag_search/sample_queries.txt}"
    if [ ! -f "$QUERY_FILE" ]; then
        echo "[setup] 生成默认 query 列表: $QUERY_FILE"
        cat > "$QUERY_FILE" <<EOF
分布式训练
FlashAttention
序列并行
BF16 混合精度
QLoRA
知识图谱
EOF
    fi
    echo "[run] trace_compare_rerank_multiplier.py --query-list $QUERY_FILE"
    python3 tests/graph_rag_search/trace_compare_rerank_multiplier.py --query-list "$QUERY_FILE"
elif [ "$1" == "--compare" ] || [ "$1" == "-c" ]; then
    shift
    QUERY="${1:-分布式训练}"
    echo "[run] trace_compare_rerank_multiplier.py --query \"$QUERY\""
    python3 tests/graph_rag_search/trace_compare_rerank_multiplier.py --query "$QUERY"
elif [ "$1" == "--trace" ] || [ "$1" == "-t" ]; then
    shift
    QUERY="${1:-分布式训练}"
    echo "[run] trace_rag.py --query \"$QUERY\""
    python3 tests/graph_rag_search/trace_rag.py --query "$QUERY"
elif [ "$1" == "--all" ] || [ "$1" == "-a" ]; then
    echo "[run] trace_rag.py 默认 query"
    python3 tests/graph_rag_search/trace_rag.py
    echo ""
    for q in "${DEFAULT_QUERIES[@]}"; do
        echo ""
        echo "[run] trace_compare_rerank_multiplier.py --query \"$q\""
        python3 tests/graph_rag_search/trace_compare_rerank_multiplier.py --query "$q"
    done
else
    echo "Usage: $0 [mode] [args]"
    echo ""
    echo "Modes:"
    echo "  -t, --trace [query]             单次分阶段 trace (默认: 分布式训练)"
    echo "  -c, --compare [query]           rerank_multiplier 1.0 vs 2.0 对比"
    echo "  -b, --compare-batch [file]      批量 query 对比 (默认: tests/graph_rag_search/sample_queries.txt)"
    echo "  -a, --all                       跑全套 (1 次 trace + 4 次对比)"
    echo ""
    echo "Examples:"
    echo "  $0 --trace \"FlashAttention\""
    echo "  $0 --compare \"序列并行\""
    echo "  $0 --compare-batch my_queries.txt"
    echo "  $0 --all"
fi
