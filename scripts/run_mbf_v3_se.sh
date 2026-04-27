#!/bin/bash

# Navigate to project root (relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Training script for mbf_v3_se (Phase 2: mbf_v3 + SE modules)
# Expected to be started after Phase 1 (mbf_v3) completes

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29523 --nproc_per_node=4 train_v2_prune.py \
    --config configs/glint360k_mbf_v3_se.py
