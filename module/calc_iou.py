#!/usr/bin/python
#coding: utf-8

import numpy as np
import torch

def calc_IoU(pred_label,gt_label, num_classes):
    pred_label = pred_label.flatten()
    gt_label = gt_label.flatten()
    confusion = confusion_matrix(num_classes, pred_label, gt_label)
    iou = confusion.diag() / confusion.sum(1)
    #miou = torch.mean(iou)
    
    return iou

def confusion_matrix(nb_classes, preds, label):
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    for t, p in zip(label.view(-1), preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    return confusion_matrix