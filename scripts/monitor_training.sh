#!/bin/bash

# Navigate to project root (relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 训练监控脚本 - 每 30 分钟检查一次训练状态并更新实验日志

REMOTE_HOST="S1-2"
LOG_FILE="/home/M40/face_rec/arcface_torch_5max/EXPERIMENT_LOG.md"
CHECKPOINT_DIR="/path/to/checkpoint"

echo "=========================================="
echo "训练监控报告 - $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 1. 检查远程训练进程
echo ""
echo "[1/5] 检查训练进程状态..."
PROCESS_COUNT=$(ssh $REMOTE_HOST "ps aux | grep 'train_v2_prune.*mbf_v3' | grep -v grep | wc -l" 2>/dev/null)
if [ "$PROCESS_COUNT" -gt 0 ]; then
    echo "✓ 训练进程运行中 ($PROCESS_COUNT 个进程)"
else
    echo "✗ 警告: 训练进程未运行!"
    exit 1
fi

# 2. 检查 GPU 利用率
echo ""
echo "[2/5] GPU 使用情况:"
ssh $REMOTE_HOST "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader" 2>/dev/null | head -4 | while IFS=',' read -r idx util mem_used mem_total temp; do
    echo "  GPU $idx: $util, Memory: $mem_used /$mem_total, Temp:$temp"
done

# 3. 获取最新训练日志
echo ""
echo "[3/5] 最新训练指标 (最后 3 条):"
ssh $REMOTE_HOST "tail -200 $CHECKPOINT_DIR/training.log | grep -E 'Speed.*Loss.*LearningRate' | tail -3" 2>/dev/null || echo "  暂无日志输出"

# 4. 检查验证结果 (如果有)
echo ""
echo "[4/5] 最新验证结果:"
ssh $REMOTE_HOST "tail -100 $CHECKPOINT_DIR/training.log | grep -E 'lfw|agedb|cfp' | tail -5" 2>/dev/null || echo "  尚未进行验证"

# 5. 检查已保存的 checkpoint
echo ""
echo "[5/5] Checkpoint 状态:"
CHECKPOINT_FILES=$(ssh $REMOTE_HOST "ls -lh $CHECKPOINT_DIR/*.pt 2>/dev/null | wc -l" 2>/dev/null)
if [ "$CHECKPOINT_FILES" -gt 0 ]; then
    echo "  已保存 $CHECKPOINT_FILES 个 checkpoint 文件"
    ssh $REMOTE_HOST "ls -lh $CHECKPOINT_DIR/*.pt 2>/dev/null | tail -3"
else
    echo "  尚未保存 checkpoint"
fi

echo ""
echo "=========================================="
echo "监控完成 - $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
