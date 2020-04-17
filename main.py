#!/usr/bin/python
# -*- Coding: utf-8 -*-

"""
平成ライダーをクラス分類するコード
"""

#ライブラリのインポート
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from module.riderlist import RIDER_LIST
from module.train_val import train, validation, test
from module.dataset import RiderDataset
from module.get_model import get_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=int, default=0, help="使用GPU番号 空き状況はnvidia-smiで調べる")
    parser.add_argument("-e", "--epoch", type=int, default=100, help="データセット周回数")
    parser.add_argument("-b", "--batchsize", type=int, default=16, help="ミニバッチサイズ")
    parser.add_argument("-lr", "--learningrate", type=float, default=0.001, help="学習率")
    parser.add_argument("-s", "--split", type=int, default=1, help="Train-Split")
    parser.add_argument("-m", "--model", type=str, default="VGG16",
                        choices=["VGG16", "ResNet101"])
    parser.add_argument("-o", "--optimizer", type=str, default="SGD",
                        choices=["SGD", "Adam"])
    parser.add_argument("--scheduler", action="store_true", help="Use Scheduler")
    parser.add_argument("--step", type=int, default=10, help="schedulerのStep(何Epoch毎に減衰させるか)")
    parser.add_argument("--num-worker", type=int, default=4, help="CPU同時稼働数 あまり気にしなくてよい")
    parser.add_argument("--modelname", type=str, default="bestmodel.pth", help="保存モデル名")
    parser.add_argument("--csv_path", type=str, default="rider-dataset/splits", help="csv_splitまでのパス")
    parser.add_argument("--root_dir", type=str, default="rider-dataset", help="データセットまでのパス")
    parser.add_argument("--save_dir", type=str, default="results", help="データセットまでのパス")

    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.gpu))   #GPUの設定

    #画像を正規化する関数の定義
    """
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip()])
    """
    transform = None

    """
    データセットの読み込み 初回はデータセットをダウンロードするためネットにつながっている必要あり
    train=Trueなら学習用画像5万枚読み込み
    train=Falseならテスト用画像1万枚読み込み
    batchsizeは学習時に一度に扱う枚数
    shuffleは画像をランダムに並び替えるかの設定 test時はオフ
    num_workersはCPU使用数みたいなもの 気にしないで良い
    """
    trainset = RiderDataset(csv_file=os.path.join(args.csv_path,
                            "train_{}.csv".format(args.split)),
                            root_dir=args.root_dir,
                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize,
                                            shuffle=True, num_workers=args.num_worker)

    valset = RiderDataset(csv_file=os.path.join(args.csv_path,
                            "val_{}.csv".format(args.split)),
                            root_dir=args.root_dir,
                            transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batchsize,
                                            shuffle=False, num_workers=args.num_worker)
    
    testset = RiderDataset(csv_file=os.path.join(args.csv_path,
                            "test_{}.csv".format(args.split)),
                            root_dir=args.root_dir,
                            transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize,
                                            shuffle=False, num_workers=args.num_worker)

    classes = RIDER_LIST
    
    #ネットワークの定義
    #net(input)とすることで画像をネットワークに入力できる
    net = get_model(args, len(classes))
    net = net.to(device)    #modelをGPUに送る

    """
    ここから誤差関数の定義
    """

    #SoftmaxCrossEntropyLossを使って誤差計算を行う。計算式はググってください。
    criterion = nn.CrossEntropyLoss()
    #学習器の設定 lr:学習率
    if args.optimizer == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=args.learningrate, momentum=0.9)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=args.learningrate)
    else:
        print("Cant set optimizer")
        exit()
    
    if args.scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)

    """
    ここから訓練ループ
    epoch       :同じデータに対し繰り返し学習を行う回数。
    """
    max_acc = 0.0

    for epoch in range(args.epoch):
        train(net, trainloader, optimizer, device, criterion, epoch, args)
        val_accuracy = validation(net, valloader, device, criterion, args)
        if args.scheduler:
            scheduler.step()

        #validationの成績が良ければモデルを保存
        if val_accuracy > max_acc:
            torch.save(net.state_dict(), os.path.join(args.save_dir, args.modelname))

    print('Finished Training')

    """
    ここからは学習したモデルで出力の確認
    """
    net.load_state_dict(torch.load(args.modelname))

    test(net, device, testloader, args, classes)

if __name__ == "__main__":
    main()