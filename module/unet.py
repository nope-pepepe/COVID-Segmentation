#!/usr/bin/python
# -*- Coding: utf-8 -*-

import torch
import torch.nn as nn

import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet
from squeeze_and_excitation.squeeze_and_excitation import ChannelSpatialSELayer as scSE

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, dropout=False, use_scSE=False):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        if dropout:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )
        
        elif use_scSE:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
                scSE(mid_ch, reduction_ratio=8),
                nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_ch, out_ch, dropout=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch, out_ch, conv_in_ch=None,  bilinear=False, dropout=True, use_scSE=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of ch
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch , in_ch // 2, kernel_size=2, stride=2)
            if conv_in_ch is not None:
                # 数を無理やり合わせる用(Efficient UNet)
                self.conv = DoubleConv(conv_in_ch, out_ch, dropout=dropout, use_scSE=use_scSE)
      
            else:
                # 普通のUNet
                self.conv = DoubleConv(in_ch, out_ch, dropout=dropout, use_scSE=use_scSE)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_ch, n_classes, bilinear=False, dropout=True, use_scSE=False, use_gain=False):
        super(UNet, self).__init__()
        self.n_ch = n_ch
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_gain = use_gain

        self.inc = DoubleConv(n_ch, 64, dropout=dropout)
        self.down1 = Down(64, 128, dropout=dropout)
        self.down2 = Down(128, 256, dropout=dropout)
        self.down3 = Down(256, 512, dropout=dropout)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, dropout=dropout)
        self.up1 = Up(1024, 512 // factor, bilinear=bilinear, dropout=dropout, use_scSE=use_scSE)
        self.up2 = Up(512, 256 // factor, bilinear=bilinear, dropout=dropout, use_scSE=use_scSE)
        self.up3 = Up(256, 128 // factor, bilinear=bilinear, dropout=dropout, use_scSE=use_scSE)
        self.up4 = Up(128, 64, bilinear=bilinear, dropout=dropout, use_scSE=use_scSE)
        self.outc = OutConv(64, n_classes)
        self.classification = OutConv(64, n_classes-1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if self.use_gain:
            classify = F.adaptive_avg_pool2d(self.classification(x), (1, 1)).squeeze()
            return {"out": logits, "class":classify}
        else:
            return {"out":logits}

class EfficientUNet(EfficientNet):
    def __init__(self, blocks_args=None, global_params=None, model_name="efficientnet-b4",
     bilinear=False, dropout=False, n_classes=4, use_scSE=False):
        
        super().__init__(blocks_args, global_params)
        self.downSamplingLayer = self._getDownSamplingLayer(model_name)

        self.up1 = Up(448, 272, conv_in_ch=384, bilinear=bilinear, dropout=dropout, use_scSE=use_scSE)
        self.up2 = Up(272, 112, conv_in_ch=192, bilinear=bilinear, dropout=dropout, use_scSE=use_scSE)
        self.up3 = Up(112, 56, conv_in_ch=88, bilinear=bilinear, dropout=dropout, use_scSE=use_scSE)
        self.up4 = Up(56, 32, conv_in_ch=52, bilinear=bilinear, dropout=dropout, use_scSE=use_scSE)

        self.up_last = Up(32, 32, conv_in_ch=17, bilinear=bilinear, dropout=dropout, use_scSE=use_scSE)

        self.outc = OutConv(32, n_classes)

    def _getDownSamplingLayer(self, model_name):
        if model_name == "efficientnet-b4":
            # DownSamplingするMBConvの一つ前の場所を返す(0start)
            return [1, 5, 9, 21]
        else:
            raise Exception("efficientnet-b4以外未対応です")

    def forward(self, inputs):
        # skip connection用レイヤを貯めるところ
        hidden_layers = []
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

            if idx in self.downSamplingLayer:
                hidden_layers.append(x)
        # UNetのDecoder部分
        for idx, h in enumerate(reversed(hidden_layers)):
            # リストにすると動かないので泣く泣くこの形に 誰かいい方法教えてください
            if idx == 0:
                x = self.up1(x, h)
            elif idx == 1:
                x = self.up2(x, h)
            elif idx == 2:
                x = self.up3(x, h)
            elif idx == 3:
                x = self.up4(x, h)        
        x = self.up_last(x, inputs)

        logits = self.outc(x)
        return {"out":logits}

    @classmethod
    def from_name(cls, model_name, in_channels=3, n_classes=4, use_scSE=False, **override_params):
        from efficientnet_pytorch.utils import get_model_params
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(blocks_args, global_params, n_classes=n_classes, use_scSE=use_scSE)
        model._change_in_channels(in_channels)
        return model

if __name__ == "__main__":
    model = UNet(1, 4, use_gain=True)
    print(model)
    from torchsummary import summary
    model = EfficientUNet.from_name("efficientnet-b4", in_channels=1, n_classes=4, use_scSE=True)
    img = torch.rand(2,1,512,512)
    #model = UNet(1, 4, use_scSE=False)
    #model(img)
    print(model)
    #print(model)
    #print(model.hoge)
    #print(summary(model, (1,512,512), device="cpu"))
