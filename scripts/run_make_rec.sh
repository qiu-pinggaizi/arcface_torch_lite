#!/bin/bash

# Navigate to project root (relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"


# 设置路径
CODE_ROOT="/ipcdata-bj/data/jinj/WebFace260M"  # 代码保存路径
DATA_ROOT="/ipcdata-bj/data/jinj/WebFace260M/WebFace260M/WebFace260M"  # 数据路径
OUTPUT_PREFIX="train"

# 进入数据目录（lst/rec 文件生成在这里）
cd "$DATA_ROOT"

echo "Step 1: Creating train.lst..."
python "$CODE_ROOT/fast_im2rec.py" \
    --list \
    --recursive \
    "$OUTPUT_PREFIX" \
    "$DATA_ROOT"

echo ""
echo "Step 2: Creating shuffled .rec and .idx files..."
python "$CODE_ROOT/fast_im2rec.py" \
    --num-thread 16 \
    --quality 100 \
    "$OUTPUT_PREFIX" \
    "$DATA_ROOT"

echo ""
echo "Step 3: Verifying generated files..."
ls -lh ${OUTPUT_PREFIX}.*

echo ""
echo "Done! Generated files:"
echo "- ${OUTPUT_PREFIX}.lst"
echo "- ${OUTPUT_PREFIX}.rec"
echo "- ${OUTPUT_PREFIX}.idx"
