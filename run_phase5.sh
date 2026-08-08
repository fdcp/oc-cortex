#!/usr/bin/env bash
# ============================================================
# run_phase5.sh — 顺序执行 P5 的 triple / entity 两种模式
# ============================================================
# 用法:
#   bash run_phase5.sh                  # 断点续传 (默认)
#   bash run_phase5.sh --force          # 强制重抽, 透传 --force 给两次 python
# 可选环境变量:
#   P5_CONFIG      配置文件路径 (默认 config/code_p5_config.yaml)
#   QDRANT_PATH    嵌入式 Qdrant 存储目录 (默认 ./qdrant_data, 跟 config/code_p5_config.yaml 一致)
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

# ---- preflight: 清理嵌入式 Qdrant 残留的 stale lock ----
# 嵌入式 Qdrant 不支持并发访问, portalocker 用 .lock 文件做互斥。
# 进程异常退出时锁文件可能残留, 下次启动 portalocker 会拒绝 (AlreadyLocked)。
# 这里只清 "无进程占用" 的 stale lock; 真有进程占着则报错让用户处理, 避免误删正在用的锁。
QDRANT_PATH="${QDRANT_PATH:-./qdrant_data}"
LOCK_FILE="$QDRANT_PATH/.lock"
if [[ -f "$LOCK_FILE" ]]; then
    if ! lsof "$LOCK_FILE" >/dev/null 2>&1; then
        echo "⚠️  发现残留 lock (无进程占用): $LOCK_FILE"
        echo "   自动清理中..."
        rm -f "$LOCK_FILE"
        echo "   ✓ 已清理"
    else
        echo "✗ $LOCK_FILE 被其他进程占用, 请先关闭后重试"
        lsof "$LOCK_FILE" || true
        exit 1
    fi
fi

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
