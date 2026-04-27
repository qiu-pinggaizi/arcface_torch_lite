import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import numpy as np
import torch

from backbones import get_model
from utils.utils_config import get_config
from utils.utils_callbacks import CallBackVerification


@torch.no_grad()
def inference(config, weight, img):

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
    net.eval()

    callback_verification(1, net)


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Validation')
    parser.add_argument('--config', type=str, default='configs/glint360k_r34.py')
    parser.add_argument('--network', type=str, default='r34', help='backbone network')
    parser.add_argument('--weight', type=str, required=True, help='path to model.pt')
    parser.add_argument('--img', type=str, default=None)
    args = parser.parse_args()
    inference(args.config, args.weight, args.img)
