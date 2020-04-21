#!/usr/bin/python
# -*- Coding: utf-8 -*-

import os

class LogIoU:
    def __init__(self, savedir, name="val"):
        self.filename = os.path.join(savedir, "{}.csv".format(name))
    
    def __call__(self, epoch, miou, iou):
        sIoU = [str(i) for i in (iou)]
        string = '%d\t%f\t'%(epoch,miou)+'\t'.join(sIoU) + '\n'
        with open(self.filename, mode="a") as f:
            f.write(string)