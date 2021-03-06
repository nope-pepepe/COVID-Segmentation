#!/usr/bin/python
# -*- Coding: utf-8 -*-

import argparse

def train_opts():
    parser = argparse.ArgumentParser()

    # 基本的な学習設定
    parser.add_argument("-g", "--gpu", type=int, default=0, help="使用GPU番号 空き状況はnvidia-smiで調べる")
    parser.add_argument("-e", "--epoch", type=int, default=100, help="データセット周回数")
    parser.add_argument("-b", "--batchsize", type=int, default=4, help="ミニバッチサイズ")
    parser.add_argument("-lr", "--learningrate", type=float, default=0.001, help="学習率")
    parser.add_argument("-m", "--model", type=str, default="Deeplab",
                        choices=["Deeplab", "UNet", "EfficientDeeplab", "EfficientUNet"])
    parser.add_argument("-o", "--optimizer", type=str, default="SGD",
                        choices=["SGD", "Adam", "RMSprop"])
    parser.add_argument("-l", "--loss", type=str, default="CE",
                        choices=["CE", "focal"])
    parser.add_argument("--scheduler", action="store_true", help="Use Scheduler")
    parser.add_argument("--gamma", type=float, default=0.1, help="schedulerの削減率")
    parser.add_argument("--step", type=int, default=10, help="schedulerのStep(何Epoch毎に減衰させるか)")
    parser.add_argument("-pre", "--pretrained", action="store_true", help="Use Pretrained model")
    parser.add_argument("--weight", action="store_false", help="Use class weight")
    parser.add_argument("--num-worker", type=int, default=4, help="CPU同時稼働数 あまり気にしなくてよい")
    parser.add_argument("--modelname", type=str, default="bestmodel.pth", help="保存モデル名")
    parser.add_argument("--root_dir", type=str, default="dataset", help="データセットまでのパス")
    parser.add_argument("--save_dir", type=str, default="results", help="保存ディレクトリまでのパス")
    
    # Class Weightの応用設定(基本的に使わない)
    parser.add_argument("--weight-softmax", action="store_true", help="Use class weight with softmax")
    parser.add_argument("--control-weight", action="store_true", help="Use class weight control to softmax value")
    
    # DropoutをUNetに適用する設定(基本的に使わない)
    parser.add_argument("--dropout", action="store_true", help="Use Dropout with UNet")

    # GAINに関する設定(基本的に使わない mIoUも下がった)
    parser.add_argument("--use-gain", action="store_true", help="Use GAIN")
    parser.add_argument("--use-mask", action="store_true", help="GAIN時に入力画像をマスクするか否か maskしない場合はSoftMarginLossになる")
    parser.add_argument("--use-softmask", action="store_true", help="softmaskするかガッツリmaskするか")
    parser.add_argument("--lambda1", type=float, default=1.0, help="GAINにおける cl lossの係数 論文ではデフォルト1.0")
    parser.add_argument("--alpha", type=float, default=1.0, help="GAINにおける attention mining lossの係数 論文ではデフォルト1.0")
    parser.add_argument("--omega", type=float, default=10.0, help="GAINにおける segmentation lossの係数 論文ではデフォルト10.0")
    
    # EfficientNetのBackbone設定 ほぼ機能していなけれど
    parser.add_argument("--backbone", type=str, default="efficientnet-b4", help="EfficientDeeplabのバックボーン選択")
    
    # scSEブロックを使うかの設定 mIoU下がったので入れなくていいと思う
    parser.add_argument("--use-scse", action="store_true", help="Use scSE")

    args = parser.parse_args()

    return args

def demo_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=int, default=0, help="使用GPU番号 空き状況はnvidia-smiで調べる")
    parser.add_argument("-b", "--batchsize", type=int, default=1, help="ミニバッチサイズ")
    parser.add_argument("-m", "--model", type=str, default="Deeplab",
                        choices=["Deeplab", "UNet", "EfficientDeeplab", "EfficientUNet"])
    parser.add_argument("-r", "--readdir", type=str, default=None, help="読み出すディレクトリ名")
    parser.add_argument("--num-worker", type=int, default=4, help="CPU同時稼働数 あまり気にしなくてよい")
    parser.add_argument("--modelname", type=str, default="bestmodel.pth", help="保存モデル名")
    parser.add_argument("--root_dir", type=str, default="dataset", help="データセットまでのパス")
    parser.add_argument("--save_dir", type=str, default="results", help="データセットまでのパス")
    parser.add_argument("-pre", "--pretrained", action="store_true", help="Use Pretrained model")
    parser.add_argument("--use-gain", action="store_true", help="Use GAIN")
    
    parser.add_argument("--backbone", type=str, default="efficientnet-b4", help="EfficientDeeplabのバックボーン選択")
    parser.add_argument("--use-scse", action="store_true", help="Use scSE")
  
    args = parser.parse_args()

    args.dropout = False

    return args