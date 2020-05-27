#!/usr/bin/python
# -*- Coding: utf-8 -*-

import os

class LogIoU:
    def __init__(self, savedir, name="val"):
        self.filename = os.path.join(savedir, "{}.csv".format(name))
    
    def __call__(self, epoch, miou, iou):
        sIoU = [str(i.item()) for i in (iou)]
        string = '%d\t%f\t'%(epoch,miou)+'\t'.join(sIoU) + '\n'
        with open(self.filename, mode="a") as f:
            f.write(string)

class LogGAINLoss:
    # GAINのLoss割合を記録していくモジュール
    def __init__(self, savedir, name="val_loss"):
        self.filename = os.path.join(savedir, "{}.csv".format(name))
    
    def __call__(self, epoch, sample):
        loss = sample["loss"]
        l_cl = sample["l_cl"]
        l_am = sample["l_am"]
        l_seg = sample["l_seg"]

        string = "{:0=3}\t{:.4f}\t{:.4f}\t{:.1f}\t{:.4f}\t{:.1f}\t{:.4f}\t{:.1f}\n".format(
            epoch, loss,
            l_cl, l_cl/loss*100,
            l_am, l_am/loss*100,
            l_seg, l_seg/loss*100
        )
        with open(self.filename, mode="a") as f:
            f.write(string)

def logOption(savedir, args):
    with open(os.path.join(savedir, "opts.csv"), mode="w") as f:
        opts_list = dir(args)
        opts_list = [s for s in opts_list if not s.startswith("_")]
        for opt in opts_list:
            setting = getattr(args, opt)
            string = "{},{}\n".format(opt, setting)
            f.write(string)