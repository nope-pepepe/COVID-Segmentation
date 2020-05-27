#!/usr/bin/python
# -*- Coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn as nn

from module.calc_iou import calc_IoU

class SumLoss:
    def __init__(self):
        self.running_loss = 0.
    
    def __call__(self, loss):
        self.running_loss += loss

def mask(pred, inputs, labels):
    inputs[pred.unsqueeze(1)==labels.unsqueeze(1)] = 0
    return inputs

def softmask(inputs, image, omega=100, sigma=0.25):
    inputs_min = inputs.min()
    inputs_max = inputs.max()
    scaled_inputs = (inputs - inputs_min) / (inputs_max - inputs_min)

    mask = F.sigmoid(omega * (scaled_inputs - sigma))

    hoge = image*mask 
    print(hoge[0][0][0][0], hoge[0][1][0][0], hoge[0][2][0][0])
    masked_image = image - image * mask
    
    return masked_image

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
            #mask_inputs = softmask(inputs=segment_outputs_1st, image=inputs)
            miningloss = model(mask_inputs)["class"].sigmoid().mean()

        if not val:
            model.train()
        segment_outputs_2nd = model(inputs)["out"]
        segment_loss = criterion(segment_outputs_2nd, segment_labels)

        l_cl = args.lambda1 * classloss
        l_am = args.alpha * miningloss
        l_seg = args.omega * segment_loss
        loss = l_cl + l_am + l_seg

        outputs = segment_outputs_2nd

        sample = {"loss":loss, "l_cl":l_cl, "l_am":l_am, "l_seg":l_seg, "outputs":outputs}

    else:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)["out"]
        loss = criterion(outputs, labels)
        sample = {"loss":loss, "outputs":outputs}
    
    return sample

def train(model, trainloader, optimizer, device, criterion, epoch, args):
    model.train()   #model訓練モードへ移行
    running_loss = 0.0  #epoch毎の誤差合計

    for i, data in enumerate(trainloader):
        #出力計算
        sample = calc_loss(data, model, criterion, device, args)
        loss = sample["loss"]
        
        #勾配計算とOptimStep
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print("Train Epoch:{:>3} Loss:{:.4f}".format(epoch, running_loss))

def validation(model, valloader, device, criterion, args):
    model.eval()    #モデル推論モードへ移行
    total_loss = SumLoss()  #epoch毎の誤差合計
    
    loss_cl = SumLoss() #GAINのみで使う
    loss_am = SumLoss() #GAINのみで使う
    loss_seg = SumLoss() #GAINのみで使う

    with torch.no_grad():   #勾配計算を行わない状態
        for i, data in enumerate(valloader):
            sample = calc_loss(data, model, criterion, device, args, val=True)

            loss = sample["loss"]
            outputs = sample["outputs"]
            if args.use_gain:
                loss_cl(sample["l_cl"].item())
                loss_am(sample["l_am"].item())
                loss_seg(sample["l_seg"].item())

            #iou計算
            _, predicted = torch.max(outputs.data, 1)
            labels = data[1]
            if i == 0:
                preds = predicted
                gt = labels
            else:
                preds = torch.cat([preds, predicted], dim=0)
                gt = torch.cat([gt, labels], dim=0)

            # running loss計算
            total_loss(loss.item())

        iou = calc_IoU(preds, gt, num_classes=args.num_classes)
        miou = torch.mean(iou)

        return_data = {"iou":iou, "miou":miou,
                       "loss":total_loss.running_loss,
                       "l_cl":loss_cl.running_loss,
                       "l_am":loss_am.running_loss,
                       "l_seg":loss_seg.running_loss}

        print("mIoU:{:3.1f}% Loss:{:.4f}".format(miou*100, total_loss.running_loss))

    return return_data