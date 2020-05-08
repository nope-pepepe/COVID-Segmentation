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
    def __init__(self, mode="train", root_dir="dataset", transform=None, channel=1, mask_img=False):
        """
        mode     : train or val
        root_dir : 画像までのパス
        transform: 変換方法
        channel  : チャンネル数
        mask_img : GAIN用マスクをするか否か(した場合ラベルは画素単位ではなく画像単位になる)
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
            label = label.transpose(2, 0, 1).astype(np.int64)

        self.img, self.label = self._ChangeImgShape(img, label, channel)

        if mask_img:
            self.img, self.label = self._mask(self.img, self.label)

        print("loaded {} images!".format(len(self.img)))

    def _ChangeImgShape(self, img, label, channel):
        img = img.transpose(2, 0, 1) # 512,512,100 → 100, 512, 512

        img_list = []
        for i in range(len(img)):
            imgArray = []
            for _ in range(channel):
                imgArray.append(img[i])
            img_list.append(imgArray)
        img_list = np.asarray(img_list)

        img_list = min_max(img_list).astype(np.float32)

        img_list, label = self._rot(img_list, label)

        return img_list, label

    def _rot(self, img, label):
        imgArray = img
        labelArray = label

        for i in range(len(img)):
            for j in range(len(imgArray[0])):
                imgArray[i, j] = np.fliplr(np.rot90(img[i, j], k=3))
    
            labelArray[i] = np.fliplr(np.rot90(label[i], k=3))

        return imgArray, labelArray

    def _mask(self, imgArray, label):
        maskArray = []
        labelArray = [] # 1 or 2 or 3 画像に対するクラスラベルが入る
        for i in range(len(imgArray)):
            for classlabel in range(1, 4):
                img = imgArray[i]
                # 該当クラスが画像内にあるか確認 胸水とか無いやつがあるため
                if np.any(label[i] == classlabel):
                    # 該当クラス以外0でマスクされた画像を作成
                    # 胸水クラスなら統合・すりガラス部分が0
                    for x in range(imgArray.shape[1]):
                        for y in range(imgArray.shape[2]):
                            if label[i, x, y] != classlabel:
                                img[x, y] = 0
                    maskArray.append(img)
                    labelArray.append(classlabel-1)
        
        maskArray = np.asarray(maskArray)
        labelArray = np.asarray(labelArray)

        return maskArray, labelArray

    def get_weight(self, softmax=False):
        num_classes = self.label.max()+1
        weights = torch.zeros(num_classes)
        for i in range(num_classes):
            num_pix = np.count_nonzero(self.label==i)
            weights[i] = num_pix

        weights /= weights.min()
        weights = 1 / weights
        if softmax:
            weights = torch.nn.functional.softmax(weights, dim=0)
        print("class weight:" + str(weights))
        return weights

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
    dataset = CovidDataset(mode="train", root_dir="../dataset/", channel=3, mask_img=True)