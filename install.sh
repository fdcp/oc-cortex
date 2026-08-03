#!/usr/bin/env bash
# ============================================================
# install.sh — OpenCode Session Knowledge Graph 安装脚本
# ============================================================
# 用法:
#   bash install.sh          # 完整安装
#   bash install.sh --light  # 轻量安装 (跳过 Reranker/Embedding 模型下载)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  OpenCode Session Knowledge Graph — Install"
echo "============================================"
echo ""

# ---- 1. Python 版本检查 ----
echo "[1/6] 检查 Python 版本..."
PYTHON_CMD=""
for cmd in python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_CMD="$cmd"
            echo "  ✓ $cmd ($ver)"
            break
        fi
    fi
done
if [ -z "$PYTHON_CMD" ]; then
    echo "  ✗ 需要 Python >= 3.10"
    echo "    安装: brew install python@3.11"
    exit 1
fi

# ---- 2. 虚拟环境 ----
echo ""
echo "[2/6] 创建虚拟环境..."
if [ ! -d ".venv" ]; then
    $PYTHON_CMD -m venv .venv
    echo "  ✓ .venv 已创建"
else
    echo "  ✓ .venv 已存在"
fi
source .venv/bin/activate

# ---- 3. pip 升级 ----
echo ""
echo "[3/6] 升级 pip..."
pip install --upgrade pip --quiet

# ---- 4. 安装依赖 ----
echo ""
echo "[4/6] 安装 Python 依赖..."

LIGHT_MODE=false
if [ "${1:-}" = "--light" ]; then
    LIGHT_MODE=true
    echo "  (轻量模式: 跳过 torch/transformers)"
fi

# 基础依赖 (Phase 1-3, 5, MCP)
# 注:含 `>=` 的包名必须加引号,否则 shell 把 `>` 当重定向,
#   会把版本号当成空文件名并丢掉版本约束(见已修复的回退 bug)。
pip install --quiet \
    "loguru>=0.7.0" \
    "pyyaml>=6.0" \
    "tiktoken>=0.5.0" \
    "openai>=1.30.0" \
    "qdrant-client>=1.7.0" \
    "sentence-transformers>=2.2.0" \
    "jieba>=0.42.1" \
    "rank_bm25>=0.2.2" \
    "networkx>=3.0" \
    "pyvis>=0.3.2" \
    "mcp>=1.0.0"
echo "  ✓ 基础依赖已安装"

# Phase 4 Reranker (torch + transformers)
if [ "$LIGHT_MODE" = false ]; then
    echo "  安装 Reranker 依赖 (torch + transformers, 约 2GB)..."
    pip install --quiet \
        "transformers>=4.40.0" \
        "torch>=2.0.0"
    echo "  ✓ Reranker 依赖已安装"
else
    echo "  ⚠ 跳过 Reranker (torch + transformers)"
    echo "    如需 Phase 4 搜索, 请重新运行: bash install.sh"
fi

# ---- 5. 预下载模型 ----
echo ""
echo "[5/6] 预下载模型..."

# Embedding model (BAAI/bge-small-zh-v1.5, ~100MB)
if [ "$LIGHT_MODE" = false ]; then
    echo "  下载 Embedding 模型 (BAAI/bge-small-zh-v1.5)..."
    $PYTHON_CMD -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('BAAI/bge-small-zh-v1.5')
print('  ✓ Embedding 模型已缓存')
" 2>/dev/null || echo "  ⚠ Embedding 模型下载失败 (网络问题? 可稍后重试)"

    # Reranker model (Qwen/Qwen3-Reranker-0.6B, ~1.2GB)
    echo "  下载 Reranker 模型 (Qwen/Qwen3-Reranker-0.6B)..."
    $PYTHON_CMD -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('Qwen/Qwen3-Reranker-0.6B', trust_remote_code=True)
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-Reranker-0.6B', trust_remote_code=True)
print('  ✓ Reranker 模型已缓存')
" 2>/dev/null || echo "  ⚠ Reranker 模型下载失败 (网络问题? 可稍后重试)"
else
    echo "  ⚠ 跳过模型下载 (轻量模式)"
fi

# ---- 6. API Key 配置 ----
echo ""
echo "[6/6] 配置 API Key..."
AUTH_FILE="$HOME/.local/share/opencode/auth.json"
if [ -f "$AUTH_FILE" ]; then
    echo "  ✓ 检测到 OpenCode auth.json"
    echo "  提示: 运行前需设置环境变量:"
    echo '    export OPENCODE_ZEN_API_KEY=$(python3 -c \'
    echo '      "import json; d=json.load(open(\"$HOME/.local/share/opencode/auth.json\")); \'
    echo '      print(d[\"opencode-go\"][\"key\"])")'
else
    echo "  ⚠ 未找到 $AUTH_FILE"
    echo "    请先登录 OpenCode 获取 API Key, 或手动设置:"
    echo "    export OPENCODE_ZEN_API_KEY=your_key_here"
fi

echo ""
echo "============================================"
echo "  安装完成!"
echo "============================================"
echo ""
echo "下一步:"
echo "  source .venv/bin/activate"
echo '  export OPENCODE_ZEN_API_KEY=$(python3 -c "import json; d=json.load(open('"'"'$HOME/.local/share/opencode/auth.json'"'"')); print(d['"'"'opencode-go'"'"']['"'"'key'"'"'])")'
echo "  export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1"
echo "  python3 src/code_p1_main.py"
echo ""
echo "完整操作指南: cat README.md"
