#!/bin/bash

# Multi-dataset training launch script
# Teacher: N datasets with per-dataset PartialFC heads and weighted loss

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=12345 \
    train_multi_data.py \
    configs/glint360k_r50_multi_dataset.py
