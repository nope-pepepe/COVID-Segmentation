#!/usr/bin/python
# -*- Coding: utf-8 -*-

"""
Covid19-Segmentation
"""

#ライブラリのインポート
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import datetime

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from module.train_val import train, validation
from module.dataset import CovidDataset
from module.get_model import get_model
from module.classes import CLASSES
from module.logger import LogIoU, logOption
from module.opts import train_opts

def main():
    args = train_opts()

    device = torch.device("cuda:{}".format(args.gpu))   #GPUの設定

    classes = CLASSES
    setattr(args, "num_classes", len(classes))

    # 現在時刻の保存ディレクトリ作成
    dt_now = datetime.datetime.now()
    savedir = os.path.join(args.save_dir, 
                    dt_now.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(savedir, exist_ok=True)
    print("make save directory {}".format(savedir))
    logOption(savedir, args)

    #画像を正規化する関数の定義
    """
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip()])
    """
    transform = None

    channel = 3 if args.model == "Deeplab" else 1

    """
    batchsizeは学習時に一度に扱う枚数
    shuffleは画像をランダムに並び替えるかの設定 test時はオフ
    num_workersはCPU使用数みたいなもの 気にしないで良い
    """
    trainset = CovidDataset(mode="train",
                            root_dir=args.root_dir,
                            transform=transform,
                            channel=channel)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize,
                                            shuffle=True, num_workers=args.num_worker)

    valset = CovidDataset(mode="val",
                            root_dir=args.root_dir,
                            transform=transform,
                            channel=channel)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batchsize,
                                            shuffle=False, num_workers=args.num_worker)
    
    #ネットワークの定義
    #net(input)とすることで画像をネットワークに入力できる
    net = get_model(args, args.num_classes)
    net = net.to(device)    #modelをGPUに送る

    if args.weight:
        weight = trainset.get_weight().to(device)
    else:
        weight = None

    """
    ここから誤差関数の定義
    """

    #SoftmaxCrossEntropyLossを使って誤差計算を行う。計算式はググってください。
    if args.loss == "CE":
        criterion = nn.CrossEntropyLoss(weight=weight)
    elif args.loss == "focal":
        from module.loss import FocalLoss
        criterion = FocalLoss()
    
    #学習器の設定 lr:学習率
    if args.optimizer == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=args.learningrate, momentum=0.9)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=args.learningrate)
    else:
        print("Can't set optimizer")
        exit()
    
    if args.scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)

    """
    ここから訓練ループ
    epoch       :同じデータに対し繰り返し学習を行う回数。
    """
    max_miou = 0.0
    vallogger = LogIoU(savedir)

    for epoch in range(args.epoch):
        train(net, trainloader, optimizer, device, criterion, epoch, args)
        iou, miou = validation(net, valloader, device, criterion, args)
        if args.scheduler:
            scheduler.step()

        #validationの成績が良ければモデルを保存
        if miou > max_miou:
            torch.save(net.state_dict(), os.path.join(savedir, args.modelname))

        vallogger(epoch, miou, iou)

    print('Finished Training')

if __name__ == "__main__":
    main()