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
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )

@torch.no_grad()
def inference(config, weight, name, img):

    cfg = get_config(config)
    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec
    )

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
    net.load_state_dict(torch.load(weight))

    
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
    # parser.add_argument('--weight', type=str, default='/ipcdata-bj/data/jinj/face_rec_train_result/mbf_large/model.pt')
    parser.add_argument('--weight', type=str, default='/ipcdata-ak/zgr/insightface/models/arcface_1.onnx')
    parser.add_argument('--img', type=str, default=None)
    args = parser.parse_args()
    inference(args.config, args.weight, args.network, args.img)
