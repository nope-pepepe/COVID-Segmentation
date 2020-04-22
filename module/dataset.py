#!/usr/bin/python
# -*- Coding: utf-8 -*-

import torch
import torch.utils.data
import nibabel as nib
import os
import numpy as np

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

class CovidDataset(torch.utils.data.Dataset):
    def __init__(self, mode="train", root_dir="dataset", transform=None, channel=1):
        """
        mode     : train or val
        root_dir : 画像までのパス
        transform: 変換方法
        size     : 変換後サイズ
        """
        print("start loading {} dataset".format(mode))
        self.transform = transform
        self.mode = mode

        # 画像の読み込み
        if mode == "train" or mode == "val":
            nii_img = nib.load(os.path.join(root_dir, "tr_im.nii"))

            # ラベル読み込み
            nii_label = nib.load(os.path.join(root_dir, "tr_mask.nii"))
            label = np.asanyarray(nii_label.dataobj)
            
        elif mode == "test":
            nii_img = nib.load(os.path.join(root_dir, "val_im.nii"))

        else:
            print("You should define mode = ['train', 'val', 'test']")
            exit()

        img = np.asarray(nii_img.dataobj)

        if mode == "train":
            img = img[:, :, :70]
            label = label[:, :, :70]
        elif mode == "val":
            img = img[:, :, 70:]
            label = label[:, : ,70:]

        if mode != "test":
            self.label = label.transpose(2, 0, 1).astype(np.int64)

        self._ChangeImgShape(img, channel)

        print("loaded {} images!".format(len(self.img)))

    def _ChangeImgShape(self, img, channel):
        img = img.transpose(2, 0, 1) # 512,512,100 → 100, 512, 512

        self.img = []
        for i in range(len(img)):
            imgArray = []
            for _ in range(channel):
                imgArray.append(img[i])
            self.img.append(imgArray)
        self.img = np.asarray(self.img)

        self.img = min_max(self.img).astype(np.float32)
        print(self.img.shape)

    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
        img = self.img[idx]

        if self.transform:
            img = self.transform(img)
        
        if self.mode != "test":
            label = self.label[idx]
            return img, label

        else:
            return img

if __name__ == "__main__":
    dataset = CovidDataset(mode="test", root_dir="../dataset/", channel=3)