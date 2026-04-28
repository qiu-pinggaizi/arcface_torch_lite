#!/bin/bash

# Multi-dataset + knowledge distillation training launch script
# Teacher: ResNet-100, Student: mbf_v3_se
# Datasets: glint360k + faces_umd

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=12346 \
    train_multi_data_distill.py \
    configs/glint360k_facesumd_mbf_v3_se_distill_multi.py
