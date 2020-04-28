#!/usr/bin/python
# -*- Coding: utf-8 -*-

"""
Covid19-Segmentation
"""

#ライブラリのインポート
import numpy as np
import argparse
import os
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from module.dataset import CovidDataset
from module.get_model import get_model
from module.classes import CLASSES
from module.opts import demo_opts

def demo(model, dataloader, device, args):
    rootdir = os.path.join(args.save_dir, args.readdir)
    pred_dir = os.path.join(rootdir, "pred")
    inputs_dir = os.path.join(rootdir, "inputs")
    gt_dir = os.path.join(rootdir, "gt")
    bl_dir = os.path.join(rootdir, "blend")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(bl_dir, exist_ok=True)

    model.eval()    #モデル推論モードへ移行

    with torch.no_grad():   #勾配計算を行わない状態
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            #出力計算
            output = model(inputs)["out"][0]
            output_pred = output.argmax(0)
            
            # create a color pallette, selecting a color for each class
            """
            palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
            print(palette)
            colors = torch.as_tensor([i for i in range(args.num_classes)])[:, None] * palette
            colors = (colors % 255).numpy().astype("uint8")
            print(colors)
            """
            colors = np.asarray([[0,0,0],
                                 [255,0,0],
                                 [0,255,0],
                                 [0,0,255]]).astype("uint8")

            # 予測
            pred_img = Image.fromarray(output_pred.byte().cpu().numpy())#.resize(inputs.size())
            pred_img.putpalette(colors)
            pred_img = pred_img.convert("RGB")
            save_path = os.path.join(pred_dir, "img_{:0=3}.jpg".format(i))
            pred_img.save(save_path, quality=95)
            
            # label
            labels = labels.squeeze(0)
            gt_img = Image.fromarray(labels.byte().cpu().numpy())#.resize(inputs.size())
            gt_img.putpalette(colors)
            gt_img = gt_img.convert("RGB")
            save_path = os.path.join(gt_dir, "gt_{:0=3}.jpg".format(i))
            gt_img.save(save_path, quality=95)

            # 入力画像
            inputs = inputs.squeeze(0)
            inputs *= 255
            inputs = inputs.cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
            if inputs.shape[2] == 1:
                inputs = np.squeeze(inputs)
            inputs_img = Image.fromarray(inputs)
            inputs_img = inputs_img.convert("RGB")
            inputs_img.save(os.path.join(inputs_dir, "img_{:0=3}.jpg".format(i)), quality=95)
            
            print(inputs_img.size)
            bl_pred = Image.blend(inputs_img, pred_img, 0.5)
            bl_gt = Image.blend(inputs_img, gt_img, 0.5)
            bl_pred.save(os.path.join(bl_dir, "pred_{:0=3}.jpg".format(i)), quality=95)
            bl_gt.save(os.path.join(bl_dir, "gt_{:0=3}.jpg".format(i)), quality=95)
            print(save_path)

def main():
    args = demo_opts()
    
    if args.readdir == None:
        print("Set reading directory")
        exit()

    device = torch.device("cuda:{}".format(args.gpu))   #GPUの設定

    classes = CLASSES
    setattr(args, "num_classes", len(classes))

    channel = 3 if args.model == "Deeplab" else 1

    dataset = CovidDataset(mode="val",
                            root_dir=args.root_dir,
                            transform=None,
                            channel=channel)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize,
                                            shuffle=False, num_workers=args.num_worker)
    
    #ネットワークの定義
    net = get_model(args, args.num_classes)
    model_path = os.path.join(args.save_dir, args.readdir, args.modelname)
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)    #modelをGPUに送る

    demo(net, dataloader, device, args)

if __name__ == "__main__":
    main()