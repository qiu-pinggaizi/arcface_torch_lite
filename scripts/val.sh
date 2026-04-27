#!/bin/bash

# Navigate to project root (relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 默认 onnx 为 False
ONNX_ARG=""

# 如果第二个参数是 onnx，则设置 onnx 参数
if [ "$2" == "onnx" ]; then
    ONNX_ARG="--onnx"
fi

# 剪枝后模型验证
CUDA_VISIBLE_DEVICES=0 python tools/inference_val_prune.py \
    --config configs/glint360k_mbf_large_prune.py \
    $ONNX_ARG \
    --prune_ratio 4.0/16.0 \
    --network mbf_large  \
    --weight /path/to/pruned/model.pt