import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import numpy as np
import torch

from backbones import get_model
from utils.utils_config import get_config
from utils.utils_callbacks import CallBackVerification

import torch_pruning as tp
import onnx


def prune_model(net, img, prune_ratio):
    """对模型执行结构化剪枝"""
    for param in net.parameters():
        param.requires_grad = True

    ignored_layers = []
    for name, m in net.named_modules():
        if 'features' in name or 'conv_sep' in name:
            ignored_layers.append(m)

    pruner = tp.pruner.GroupNormPruner(
        net,
        img,
        importance=tp.importance.GroupMagnitudeImportance(),
        iterative_steps=1,
        pruning_ratio=prune_ratio,
        ignored_layers=ignored_layers
    )
    pruner.step()
    return net


def export_onnx(net, img, output_path, opset=11, simplify=True):
    """将模型导出为 ONNX 格式"""
    torch.onnx.export(
        net, img, output_path,
        input_names=["data"],
        keep_initializers_as_inputs=False,
        verbose=False,
        opset_version=opset
    )
    model = onnx.load(output_path)
    if model.ir_version > 9:
        print(f"Warning: 原始ir_version={model.ir_version}，将降级为9以兼容")
        model.ir_version = 9
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "1"
    if simplify:
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, output_path)
    print(f"ONNX model saved to {output_path}")


def get_res(config, weight, network, img, export_onnx_flag, prune_ratio):
    """加载剪枝模型并验证/导出"""

    cfg = get_config(config)

    # 准备输入
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    # 创建模型并剪枝
    net = get_model(network, fp16=False)
    net = prune_model(net, img, prune_ratio)

    # 加载剪枝后的权重
    state_dict = torch.load(weight, map_location="cpu")
    net.load_state_dict(state_dict, strict=False)

    # ONNX 导出模式
    if export_onnx_flag:
        onnx_path = weight.rsplit('.', 1)[0] + ".onnx"
        export_onnx(net, img, onnx_path)
        return

    # 验证模式
    for param in net.parameters():
        param.requires_grad = False
    net.eval()

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec
    )
    callback_verification(1, net)


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='Pruned Model Validation / ONNX Export')
    parser.add_argument('--config', type=str, default='configs/glint360k_mbf_large_prune.py')
    parser.add_argument('--network', type=str, default='mbf_large', help='backbone network')
    parser.add_argument('--weight', type=str, required=True, help='path to pruned model .pt file')
    parser.add_argument('--img', type=str, default=None)
    parser.add_argument('--onnx', action='store_true', help='export to ONNX instead of validation')
    parser.add_argument('--prune_ratio', type=float, default=0.25, help='pruning ratio (0.0 ~ 1.0)')
    args = parser.parse_args()

    get_res(args.config, args.weight, args.network, args.img, args.onnx, args.prune_ratio)
