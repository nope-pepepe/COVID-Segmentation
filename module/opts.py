#!/usr/bin/python
# -*- Coding: utf-8 -*-

import argparse

def train_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=int, default=0, help="使用GPU番号 空き状況はnvidia-smiで調べる")
    parser.add_argument("-e", "--epoch", type=int, default=100, help="データセット周回数")
    parser.add_argument("-b", "--batchsize", type=int, default=4, help="ミニバッチサイズ")
    parser.add_argument("-lr", "--learningrate", type=float, default=0.001, help="学習率")
    parser.add_argument("-m", "--model", type=str, default="Deeplab",
                        choices=["Deeplab"])
    parser.add_argument("-o", "--optimizer", type=str, default="SGD",
                        choices=["SGD", "Adam"])
    parser.add_argument("--scheduler", action="store_true", help="Use Scheduler")
    parser.add_argument("-pre", "--pretrained", action="store_true", help="Use Pretrained model")
    parser.add_argument("--step", type=int, default=10, help="schedulerのStep(何Epoch毎に減衰させるか)")
    parser.add_argument("--num-worker", type=int, default=4, help="CPU同時稼働数 あまり気にしなくてよい")
    parser.add_argument("--modelname", type=str, default="bestmodel.pth", help="保存モデル名")
    parser.add_argument("--root_dir", type=str, default="dataset", help="データセットまでのパス")
    parser.add_argument("--save_dir", type=str, default="results", help="データセットまでのパス")

    args = parser.parse_args()

    return args

def demo_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=int, default=0, help="使用GPU番号 空き状況はnvidia-smiで調べる")
    parser.add_argument("-b", "--batchsize", type=int, default=1, help="ミニバッチサイズ")
    parser.add_argument("-m", "--model", type=str, default="Deeplab",
                        choices=["Deeplab"])
    parser.add_argument("-r", "--readdir", type=str, default=None, help="読み出すディレクトリ名")
    parser.add_argument("--num-worker", type=int, default=4, help="CPU同時稼働数 あまり気にしなくてよい")
    parser.add_argument("--modelname", type=str, default="bestmodel.pth", help="保存モデル名")
    parser.add_argument("--root_dir", type=str, default="dataset", help="データセットまでのパス")
    parser.add_argument("--save_dir", type=str, default="results", help="データセットまでのパス")
    parser.add_argument("-pre", "--pretrained", action="store_true", help="Use Pretrained model")
    
    args = parser.parse_args()

    return args