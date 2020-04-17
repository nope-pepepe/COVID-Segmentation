#!/usr/bin/python
# -*- Coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
from torchvision import models

def get_model(args, num_classes):
    if args.model == "VGG16":
        model = models.vgg16_bn(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
    elif args.model == "ResNet101":
        model = models.resnet101(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model