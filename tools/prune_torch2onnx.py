import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import onnx
import torch
import torch_pruning as tp
from copy import deepcopy
from thop import profile, clever_format

def get_model_infos(model, example_inputs, infomodelstr, org_model_info = []):
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    flops, params = profile(model, inputs=(example_inputs,), verbose=False)

    # print("example_inputs_shape: ", example_inputs.shape)
    if org_model_info == []:
        print("example_inputs_shape: ", example_inputs.shape)
        print(f"{infomodelstr}: MACs={macs / 1e9: .5f} G, #Params={nparams / 1e6: .5f} M, #Flops={flops / 1e9: .5f} G")
    else:
        assert len(org_model_info) == 3
        print(f"{infomodelstr}: MACs={macs / 1e9: .3f} G({macs/org_model_info[0]: .3f}), #Params={nparams / 1e6: .3f} M ({nparams/org_model_info[1]: .3f}), #Flops={flops/1e9: .3f} G ({flops/org_model_info[2]: .3f})")

    return macs, nparams, flops

def prune_convert_onnx(net, prune_rate_dict, path_module, output, opset=11, simplify=False):
    assert isinstance(net, torch.nn.Module)
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    img = img.astype(float)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()

    weight = torch.load(path_module, map_location='cpu')
    net.load_state_dict(weight, strict=True)
    net.eval()
  
    # org_macs, org_nparams, org_flops = get_model_infos(net, img, "arcface_model")
    

    for prune_rate_base in prune_rate_dict:
        for rate in prune_rate_dict[prune_rate_base]:
            prune_net = deepcopy(net)
            
            # import pdb;pdb.set_trace()
            pruning_ratio = (prune_rate_base-rate) / (prune_rate_base * 1.0)
            prune_output = output+ "/model.prune-" + str(rate) + "_" + str(prune_rate_base) + ".onnx"


            ignored_layers= []
            from backbones.mobilefacenet import  GDC     
            for name, m in prune_net.named_modules():
                if 'features' in name or 'conv_sep' in name:
                # if isinstance(m, (GDC,)):
                    ignored_layers.append(m)
            # import pdb;pdb.set_trace()
            pruner = tp.pruner.GroupNormPruner(
                prune_net,
                img,
                importance=tp.importance.GroupMagnitudeImportance(),  # L2 norm pruning,
                iterative_steps=1,
                pruning_ratio=pruning_ratio,
                ignored_layers=ignored_layers
                # unwrapped_parameters=unwrapped_parameters
            )
            pruner.step()
            # prune_macs, prune_nparams, prune_flops = get_model_infos(prune_net, img, "arcface_model-" + str(rate) + "_" + str(prune_rate_base), [org_macs, org_nparams, org_flops])
            
            # import pdb;pdb.set_trace()
            torch.onnx.export(prune_net, img, prune_output, input_names=["data"], keep_initializers_as_inputs=False, verbose=False, opset_version=opset)
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
            del prune_net
            del pruner

    
if __name__ == '__main__':
    import os
    import argparse
    from backbones import get_model

    parser = argparse.ArgumentParser(description='ArcFace PyTorch to onnx')
    parser.add_argument('-i','--input', type=str, default="/path/to/model.pt", help='input backbone.pth file or path')
    parser.add_argument('-o','--output', type=str, default=None, help='output onnx path')
    parser.add_argument('--network', type=str, default="mbf_large", help='backbone network')
    parser.add_argument('--simplify', action='store_true', default=True, help='onnx simplify')
    parser.add_argument('--no-simplify', dest='simplify', action='store_false', help='disable onnx simplify')
    args = parser.parse_args()
    input_file = args.input
    if os.path.isdir(input_file):
        input_file = os.path.join(input_file, "model.pt")
    assert os.path.exists(input_file)
    # model_name = os.path.basename(os.path.dirname(input_file)).lower()
    # params = model_name.split("_")
    # if len(params) >= 3 and params[1] in ('arcface', 'cosface'):
    #     if args.network is None:
    #         args.network = params[2]
    assert args.network is not None
    print(args)
    backbone_onnx = get_model(args.network, dropout=0.0, fp16=False, num_features=512)
    assert os.path.isdir(args.output), args.output + " is not a directory"
    
    prune_rate_dict = {
        10:[10, 9, 8, 7, 6],
        16:[15,14,13,12,11,10,9,8,7],
    }
    
    prune_convert_onnx(backbone_onnx, prune_rate_dict, input_file, args.output, simplify=args.simplify)
