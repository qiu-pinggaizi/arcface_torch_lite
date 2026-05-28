#!/bin/bash
# WF42M 蒸馏: r50(Teacher) → mbf_v3_se(Student) (4 GPU)
# bs=256/GPU × 4 = 1024, lr=0.1, cosine distill
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port 29532 --nproc_per_node=4 \
    train_v2_distill.py --config configs/wf42m_mbf_v3_se_distill
