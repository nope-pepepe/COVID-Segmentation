#!/usr/bin/python
# -*- Coding: utf-8 -*-

import torch

def train(model, trainloader, optimizer, device, criterion, epoch, args):
    model.train()   #model訓練モードへ移行
    correct = 0
    total = 0
    running_loss = 0.0  #epoch毎の誤差合計

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        #出力計算
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        #Accuracy計算用素材
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        #勾配計算とAdamStep
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    accuracy = correct / total * 100.0
    print("Train Epoch:{:>3} Acc:{:3.1f}% Loss:{:.4f}".format(epoch, accuracy, running_loss))

def validation(model, valloader, device, criterion, args):
    model.eval()    #モデル推論モードへ移行
    correct = 0
    total = 0
    running_loss = 0.0  #epoch毎の誤差合計

    with torch.no_grad():   #勾配計算を行わない状態
        for i, (inputs, labels) in enumerate(valloader):
            inputs, labels = inputs.to(device), labels.to(device)
            #出力計算
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            #Accuracy計算用素材
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
            running_loss += loss.item()

        accuracy = correct / total * 100.0
        print("Val Acc:{:3.1f}% Loss:{:.4f}".format(accuracy, running_loss))

    return accuracy

def test(model, device, dataloader, args, classes):
    model.eval()

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(args.batchsize):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10): #各クラスのAccuracy表示
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
