#!/bin/bash
# WF42M r50 PFC-0.2 训练 (4 GPU 速度测试)
# bs=256/GPU × 4 = 1024, lr=0.2
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29531 --nproc_per_node=4 \
    train_v2.py configs/wf42m_pfc02_4gpus_r50
