import argparse
import math
import os
import copy

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

from config import Config
from models import *

parser = argparse.ArgumentParser(description='FQ-ViT')

parser.add_argument('model',
                    choices=[
                        'toy'
                    ],
                    help='model')
parser.add_argument('--data', default='/home/hujunhao/hdd-hujunhao/tiny-imagenet-200', help='path to dataset')
parser.add_argument('--ptf', default=False, action='store_true')
parser.add_argument('--lis', default=False, action='store_true')
parser.add_argument('--quant-method',
                    default='minmax',
                    choices=['minmax', 'ema', 'omse', 'percentile'])
parser.add_argument('--calib-batchsize',
                    default=100,
                    type=int,
                    help='batchsize of calibration set')
parser.add_argument('--calib-iter', default=10, type=int)
parser.add_argument('--val-batchsize',
                    default=100,
                    type=int,
                    help='batchsize of validation set')
parser.add_argument('--num-workers',
                    default=16,
                    type=int,
                    help='number of data loading workers (default: 16)')
parser.add_argument('--device', default='cpu', type=str, help='device')
parser.add_argument('--print-freq',
                    default=100,
                    type=int,
                    help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed')


def str2model(name):
    d = {
        'toy': toy
    }
    print('Model: %s' % d[name].__name__)
    return d[name]


def seed(seed=0):
    import os
    import random
    import sys

    import numpy as np
    import torch
    sys.setrecursionlimit(100000)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = parser.parse_args()
    seed(args.seed)

    device = torch.device(args.device)
    cfg = Config(args.ptf, args.lis, args.quant_method)
    model = str2model(args.model)(in_features=3 * 224 * 224, out_features=10, \
        input_quant=False, output_quant=False, cfg=cfg)
    model_quant = copy.deepcopy(model)
    model = model.to(device)
    model_quant = model_quant.to(device)

    # ViT strategies of data preprocessing.
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    crop_pct = 0.9

    train_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)
    val_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)

    # Data
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    train_dataset = datasets.ImageFolder(traindir, train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.calib_batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dataset = datasets.ImageFolder(valdir, val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # switch to evaluate mode
    model.eval()
    model_quant.eval()
        
    # Get calibration set.
    image_list = []
    for i, (data, target) in enumerate(train_loader):
        if i == args.calib_iter:
            break
        data = data.to(device)
        image_list.append(data)

    print('Calibrating...')
    model_quant.model_open_calibrate()
    with torch.no_grad():
        for i, image in enumerate(image_list):
            if i == len(image_list) - 1:
                # This is used for OMSE method to
                # calculate minimum quantization error
                model_quant.model_open_last_calibrate()
            output = model_quant(image.view(-1, 3 * 224 * 224))
    model_quant.model_close_calibrate()
    model_quant.model_quant()

    for i, (data, target) in enumerate(val_loader):
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(data.view(-1, 3 * 224 * 224))
            output_quant = model_quant(data.view(-1, 3 * 224 * 224))
        diff = torch.max(torch.abs(output - output_quant))
        print(diff)
        break
        

def build_transform(input_size=224,
                    interpolation='bicubic',
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    crop_pct=0.875):

    def _pil_interp(method):
        if method == 'bicubic':
            return Image.BICUBIC
        elif method == 'lanczos':
            return Image.LANCZOS
        elif method == 'hamming':
            return Image.HAMMING
        else:
            return Image.BILINEAR

    resize_im = input_size > 32
    t = []
    if resize_im:
        size = int(math.floor(input_size / crop_pct))
        ip = _pil_interp(interpolation)
        t.append(
            transforms.Resize(
                size,
                interpolation=ip),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


if __name__ == '__main__':
    main()
