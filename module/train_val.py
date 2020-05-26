#!/usr/bin/python
# -*- Coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn as nn

from module.calc_iou import calc_IoU

def mask(pred, inputs, labels):
    inputs[pred.unsqueeze(1)==labels.unsqueeze(1)] = 0
    return inputs

def calc_loss(data, model, criterion, device, args, val=False):
    if args.use_gain and not val:
        # Validation入力画像をGAINのままにするとマスクされた画像から評価されてしまう
        # そのためValidation時はGAIN処理を行わない
        inputs, segment_labels, class_labels = data
        inputs, segment_labels, class_labels = inputs.to(device), segment_labels.to(device), class_labels.to(device)

        with torch.no_grad():
            model.eval()
            outputs_data = model(inputs)
            segment_outputs_1st, class_outputs = outputs_data["out"], outputs_data["class"]
            if len(class_outputs.size()) == 1:
                class_outputs = class_outputs.unsqueeze(0)

            if args.use_mask:
                # mask画像から単一クラス予測
                classloss = F.cross_entropy(class_outputs, class_labels)
            else:
                # multi label soft margin loss
                classloss = F.multilabel_soft_margin_loss(class_outputs, class_labels)

            mask_inputs = mask(torch.max(segment_outputs_1st.data, 1)[1], inputs, segment_labels)
            miningloss = model(mask_inputs)["class"].sigmoid().mean()

        if not val:
            model.train()
        segment_outputs_2nd = model(inputs)["out"]
        segment_loss = criterion(segment_outputs_2nd, segment_labels)

        loss = args.lambda1 * classloss + args.alpha * miningloss + args.omega * segment_loss

        outputs = segment_outputs_2nd

    else:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)["out"]
        loss = criterion(outputs, labels)
    
    if val:
        return loss, outputs
    else:
        return loss

def train(model, trainloader, optimizer, device, criterion, epoch, args):
    model.train()   #model訓練モードへ移行
    running_loss = 0.0  #epoch毎の誤差合計

    for i, data in enumerate(trainloader):
        #出力計算
        loss = calc_loss(data, model, criterion, device, args)
        #勾配計算とOptimStep
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print("Train Epoch:{:>3} Loss:{:.4f}".format(epoch, running_loss))

def validation(model, valloader, device, criterion, args):
    model.eval()    #モデル推論モードへ移行
    running_loss = 0.0  #epoch毎の誤差合計

    with torch.no_grad():   #勾配計算を行わない状態
        for i, data in enumerate(valloader):
            loss, outputs = calc_loss(data, model, criterion, device, args, val=True)

            #iou計算
            _, predicted = torch.max(outputs.data, 1)
            labels = data[1]
            if i == 0:
                preds = predicted
                gt = labels
            else:
                preds = torch.cat([preds, predicted], dim=0)
                gt = torch.cat([gt, labels], dim=0)

            running_loss += loss.item()

        iou = calc_IoU(preds, gt, num_classes=args.num_classes)
        miou = torch.mean(iou)

        print("mIoU:{:3.1f}% Loss:{:.4f}".format(miou*100, running_loss))

    return iou, miou