#!/usr/bin/python
# -*- Coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
from torchvision import models

def get_model(args, num_classes):
    if args.model == "Deeplab":
        model = models.segmentation.deeplabv3_resnet101(
            pretrained=args.pretrained,
            num_classes=num_classes
        )
    
    else:
        print("no such a model")
        exit()
    return model