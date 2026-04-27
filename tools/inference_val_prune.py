import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import numpy as np
import torch
from torch import distributed
import onnxruntime as ort

from backbones import get_model
from utils.utils_config import get_config
from utils.utils_callbacks import CallBackLogging, CallBackVerification

import torch_pruning as tp
from copy import deepcopy
from thop import profile, clever_format

import onnx

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:15814",
        rank=rank,
        world_size=world_size,
    )

# def get_model_infos(model, example_inputs, infomodelstr, org_model_info = []):
#     macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
#     flops, params = profile(model, inputs=(example_inputs,), verbose=False)

#     # print("example_inputs_shape: ", example_inputs.shape)
#     if org_model_info == []:
#         print("example_inputs_shape: ", example_inputs.shape)
#         print(f"{infomodelstr}: MACs={macs / 1e9: .5f} G, #Params={nparams / 1e6: .5f} M, #Flops={flops / 1e9: .5f} G")
#     else:
#         assert len(org_model_info) == 3
#         print(f"{infomodelstr}: MACs={macs / 1e9: .3f} G({macs/org_model_info[0]: .3f}), #Params={nparams / 1e6: .3f} M ({nparams/org_model_info[1]: .3f}), #Flops={flops/1e9: .3f} G ({flops/org_model_info[2]: .3f})")

#     return macs, nparams, flops

def get_res(config, weight, name, img, onnx_flag, prune_ratio):

    cfg = get_config(config)
   

    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    net = get_model(cfg.network, fp16=False)
    # org_macs, org_nparams, org_flops = get_model_infos(net, img, "arcface_model")
    #####prune
    for name, param in net.named_parameters():
        param.requires_grad = True 
    ignored_layers= []
    from backbones.mobilefacenet import  GDC     
    for name, m in net.named_modules():
        if 'features' in name or 'conv_sep' in name:
            ignored_layers.append(m)
    # import pdb;pdb.set_trace()
    pruner = tp.pruner.GroupNormPruner(
        net,
        img,
        importance=tp.importance.GroupMagnitudeImportance(),  # L2 norm pruning,
        iterative_steps=1,
        pruning_ratio=prune_ratio,
        ignored_layers=ignored_layers
        # unwrapped_parameters=unwrapped_parameters
    )
    pruner.step()
    # prune_macs, prune_nparams, prune_flops = get_model_infos(net, img, "arcface_model-prune", [org_macs, org_nparams, org_flops])
    
    net.load_state_dict(torch.load(weight))
    
    # ####SAVE PRUNE ONNX
    if 1:
        prune_output = weight + ".onnx"
        simplify = True
        opset =11
        torch.onnx.export(net, img, prune_output, input_names=["data"], keep_initializers_as_inputs=False, verbose=False, opset_version=opset)
        model = onnx.load(prune_output)
        if model.ir_version > 9:
            print(f"Warning: 原始ir_version={model.ir_version}，将降级为9以兼容")
            model.ir_version = 9  # 将ir_version设置为9
        graph = model.graph
        graph.input[0].type.tensor_type.shape.dim[0].dim_param = "1"
        if simplify:
            from onnxsim import simplify
            model, check = simplify(model)
            assert check, "Simplified ONNX model could not be validated"
        onnx.save(model, prune_output)
        exit()

    
    #####
    for name, param in net.named_parameters():
        param.requires_grad = False
    inference(net, cfg, weight, name, img)


    

# @torch.no_grad()
def inference(net, cfg, weight, name, img):
    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec
    )

    # net.eval()
    # feat = net(img).numpy()
    # print(feat)

    # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    # net = ort.InferenceSession(weight, providers=providers)

    callback_verification(1, net)



if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--config', type=str, default='configs/glint360k_r34.py')
    parser.add_argument('--network', type=str, default='r34', help='backbone network')
    # parser.add_argument('--weight', type=str, default='/path/to/model.pt')
    parser.add_argument('--weight', type=str, default='/path/to/model.onnx')
    parser.add_argument('--img', type=str, default=None)
    parser.add_argument('--onnx', action='store_true')
    parser.add_argument('--prune_ratio', type=str, default=0.1)
    args = parser.parse_args()

    get_res(args.config, args.weight, args.network, args.img, args.onnx, eval(args.prune_ratio))
