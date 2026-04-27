#!/bin/bash

# Navigate to project root (relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29521 --nproc_per_node=4 train_v2_prune.py \
        --config  configs/glint360k_mbf_large_prune.py  \
        --prune_ratio 0.2  \
        --prune_load /path/to/pruned/model.pt
