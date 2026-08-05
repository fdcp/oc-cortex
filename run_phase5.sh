#!/usr/bin/env bash
# ============================================================
# run_phase5.sh — 顺序执行 P5 的 triple / entity 两种模式
# ============================================================
# 用法:
#   bash run_phase5.sh                  # 断点续传 (默认)
#   bash run_phase5.sh --force          # 强制重抽, 透传 --force 给两次 python
# 可选环境变量:
#   P5_CONFIG   配置文件路径 (默认 config/code_p5_config.yaml)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 参数解析 ----
FORCE_FLAG=""
if [[ "${1:-}" == "--force" ]]; then
    FORCE_FLAG="--force"
fi

P5_CONFIG="${P5_CONFIG:-config/code_p5_config.yaml}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  P5 Knowledge Graph — triple + entity"
echo "  config:    $P5_CONFIG"
echo "  log dir:   $LOG_DIR/"
if [[ -n "$FORCE_FLAG" ]]; then
    echo "  mode:      ⚠ FORCE (强制重抽, 将删除已有抽取结果)"
else
    echo "  mode:      断点续传 (默认; 旧结果会被复用)"
fi
echo "============================================"
echo ""

# ---- 1/2: triple 模式 ----
echo "[1/2] P5 triple 模式 启动..."
START_T=$(date +%s)
if ! python3 src/code_p5_main.py \
    --config "$P5_CONFIG" \
    --visualize --extraction_mode triple \
    $FORCE_FLAG \
    2>&1 | tee "$LOG_DIR/phase5_triple.log"; then
    echo "  ✗ triple 模式执行失败，请查看 $LOG_DIR/phase5_triple.log"
    exit 1
fi
END_T=$(date +%s)
echo "  ✓ triple 完成 (耗时 $((END_T - START_T))s)"
echo ""

# ---- 2/2: entity 模式 ----
echo "[2/2] P5 entity 模式 启动..."
START_E=$(date +%s)
if ! python3 src/code_p5_main.py \
    --config "$P5_CONFIG" \
    --visualize --extraction_mode entity \
    $FORCE_FLAG \
    2>&1 | tee "$LOG_DIR/phase5_entity.log"; then
    echo "  ✗ entity 模式执行失败，请查看 $LOG_DIR/phase5_entity.log"
    exit 1
fi
END_E=$(date +%s)
echo "  ✓ entity 完成 (耗时 $((END_E - START_E))s)"
echo ""

echo "============================================"
echo "  P5 两种模式全部完成"
echo "    - $LOG_DIR/phase5_triple.log"
echo "    - $LOG_DIR/phase5_entity.log"
echo "============================================"
