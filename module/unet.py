#!/usr/bin/python
# -*- Coding: utf-8 -*-

import torch
import torch.nn as nn

import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, dropout=True):
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

    def __init__(self, in_ch, out_ch, bilinear=False, dropout=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of ch
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch , in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)

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
    def __init__(self, n_ch, n_classes, bilinear=False, dropout=True):
        super(UNet, self).__init__()
        self.n_ch = n_ch
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_ch, 64, dropout=dropout)
        self.down1 = Down(64, 128, dropout=dropout)
        self.down2 = Down(128, 256, dropout=dropout)
        self.down3 = Down(256, 512, dropout=dropout)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, dropout=dropout)
        self.up1 = Up(1024, 512 // factor, bilinear, dropout=dropout)
        self.up2 = Up(512, 256 // factor, bilinear, dropout=dropout)
        self.up3 = Up(256, 128 // factor, bilinear, dropout=dropout)
        self.up4 = Up(128, 64, bilinear, dropout=dropout)
        self.outc = OutConv(64, n_classes)

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
        return {"out":logits}

if __name__ == "__main__":
    model = UNet(1, 4, dropout=True)
    print(model)