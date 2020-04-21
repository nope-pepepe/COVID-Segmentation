#!/usr/bin/python
# -*- Coding: utf-8 -*-

import torch
from tqdm import tqdm

from module.calc_iou import calc_IoU

def train(model, trainloader, optimizer, device, criterion, epoch, args):
    model.train()   #model訓練モードへ移行
    running_loss = 0.0  #epoch毎の誤差合計

    for i, (inputs, labels) in enumerate(tqdm(trainloader, desc="train")):
        inputs, labels = inputs.to(device), labels.to(device)
        #出力計算
        outputs = model(inputs)["out"]
        loss = criterion(outputs, labels)

        #勾配計算とOptimStep
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    tqdm.write("Train Epoch:{:>3} Loss:{:.4f}".format(epoch, running_loss))

def validation(model, valloader, device, criterion, args):
    model.eval()    #モデル推論モードへ移行
    running_loss = 0.0  #epoch毎の誤差合計

    with torch.no_grad():   #勾配計算を行わない状態
        for i, (inputs, labels) in enumerate(tqdm(valloader, desc="val")):
            inputs, labels = inputs.to(device), labels.to(device)
            #出力計算
            outputs = model(inputs)["out"]
            loss = criterion(outputs, labels)

            #iou計算
            _, predicted = torch.max(outputs.data, 1)
            if i == 0:
                preds = predicted
                gt = labels
            else:
                preds = torch.cat([preds, predicted], dim=0)
                gt = torch.cat([gt, labels], dim=0)

            running_loss += loss.item()

        iou = calc_IoU(preds, gt, num_classes=args.num_classes)
        miou = torch.mean(iou)

        tqdm.write("mIoU:{:3.1f}% Loss:{:.4f}".format(miou*100, running_loss))

    return iou, miou