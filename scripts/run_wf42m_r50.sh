#!/bin/bash
# WF42M r50 PFC-0.2 训练 (8 GPU)
# bs=256/GPU × 8 = 2048, lr=0.4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port 29530 --nproc_per_node=8 \
    train_v2.py configs/wf42m_pfc02_8gpus_r50
