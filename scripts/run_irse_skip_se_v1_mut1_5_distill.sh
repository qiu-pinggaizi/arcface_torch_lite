#!/bin/bash

# Navigate to project root (relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port 29530 --nproc_per_node=4 train_v2_distill.py \
    --config configs/glint360k_irse_skip_se_v1_mut1_5_distill.py
