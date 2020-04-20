#!/usr/bin/python
# -*- Coding: utf-8 -*-

import torch
import torch.utils.data
import nibabel as nib
import os
import numpy as np

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

        img = np.asanyarray(nii_img.dataobj)

        if mode == "train":
            img = img[:, :, :70]
            label = label[:, :, :70]
        elif mode == "val":
            img = img[:, :, 70:]
            label = label[:, : ,70:]

        self.label = label.transpose(2, 0, 1)

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
        print(self.img.shape)

    def __len__(self):
        return len(self.img[0, 0])
    
    def __getitem__(self, idx):
        img = self.img[idx]
        label = self.label[idx]

        if self.transform:
            img = self.transform(img)
        
        return img, label

if __name__ == "__main__":
    dataset = CovidDataset(mode="train", root_dir="../dataset/", channel=3)