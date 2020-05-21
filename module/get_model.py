#!/usr/bin/python
# -*- Coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from collections import OrderedDict

from module.unet import UNet, EfficientUNet

from efficientnet_pytorch import EfficientNet

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
    
    elif args.model == "EfficientDeeplab":
        model = efficient_deeplabv3(args)
    
    elif args.model == "EfficientUNet":
        model = efficient_unet(args)
    
    else:
        print("no such a model")
        exit()
    return model

class Deeplabv3(models.segmentation._utils._SimpleSegmentationModel):
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        result = OrderedDict()
        x = features
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result

class EfficientNetExtractor(EfficientNet):
    def forward(self, inputs):
        """use convolution layer to extract feature .
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of the final convolution 
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
        
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

def efficient_deeplabv3(args):
    # args.backbone には 'efficientnet-b0'などが入る
    efficientnet = EfficientNetExtractor.from_name(args.backbone, in_channels=1)

    num_ch = efficientnet._conv_head.out_channels
    classifier = DeepLabHead(num_ch, args.num_classes)
    base_model = Deeplabv3
    model = base_model(efficientnet, classifier)
    return model

def efficient_unet(args):
    # args.backbone には 'efficientnet-b0'などが入る
    model = EfficientUNet.from_name(args.backbone, in_channels=1)
    return model

if __name__ == "__main__":
    class A:
        pass
    args = A() 
    args.backbone = "efficientnet-b0"
    args.num_classes = 4
    model = efficient_deeplabv3(args)
    img = torch.rand(2,1,512,512)
    print(model(img)["out"].shape)