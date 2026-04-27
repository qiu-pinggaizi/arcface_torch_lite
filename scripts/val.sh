#!/bin/bash

# 默认 onnx 为 False
ONNX_ARG=""

# 如果第二个参数是 onnx，则设置 onnx 参数
if [ "$2" == "onnx" ]; then
    ONNX_ARG="--onnx"
fi

#IPC-arcface
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 inference_val.py --config configs/glint360k_mbf_v2.py \
# --network mbf_large_v2  --weight /ipcdata-ak/data/jiangnanfei/code/IPC/arcface-jinj/arcface_torch/face_rec_train_result/mbf_large_v2/model.pt

#5max-arcface-mobilefacenet
# CUDA_VISIBLE_DEVICES=0 python inference_val.py --config configs/glint360k_mbf_large.py \
# --network mbf_large  --weight /ipcdata-ak/data/jiangnanfei/code/IPC/arcface-jinj/arcface_torch/face_rec_train_result/mbf_large_prune/5_model.pt

#剪枝后模型val
CUDA_VISIBLE_DEVICES=0 python inference_val_prune.py \
    --config configs/glint360k_mbf_large_prune.py \
    $ONNX_ARG \
    --prune_ratio 4.0/16.0 \
    --network mbf_large  \
    --weight /ipcdata-bj/data/jinj/face_rec_train_result/mbf_large_0.4ir_prune/model.pt