#!/usr/bin/python
# -*- Coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
from torchvision import models

from module.unet import UNet

def get_model(args, num_classes):
    if args.model == "Deeplab":
        if args.pretrained:
            model = models.segmentation.deeplabv3_resnet101(pretrained=True)
            num_ftr = model.classifier[-1].in_channels
            model.classifier[-1] = nn.Conv2d(num_ftr, num_classes,
                        kernel_size=(1, 1), stride=(1, 1))

            num_ftr_aux = model.aux_classifier[-1].in_channels
            model.aux_classifier[-1] = nn.Conv2d(num_ftr_aux, num_classes,
                        kernel_size=(1, 1), stride=(1, 1))

        else:
            model = models.segmentation.deeplabv3_resnet101(
                num_classes=num_classes
            )
    elif args.model == "UNet":
        model = UNet(1, num_classes, dropout=args.dropout)
    
    else:
        print("no such a model")
        exit()
    return model