# COVID-Segmentation
## 概要
<http://medicalsegmentation.com/covid19/> の画像をセグメンテーションするためのリポジトリ

現状とりあえずDeeplabv3+(ResNet101)で動きます。

Train:70枚

Validation:30枚

で学習します。

## 学習方法
例:python3 main.py -g 0 -e 100 -b 4 -lr 0.001 -o Adam -pre

適宜書き換えてください。

BatchSizeはとりあえず4で9GBくらいメモリ食います。

メモリ並列できたら嬉しい(将来的にやるかも)

## その他
Validation時Confusion Matrix計算で1Epochあたり10分弱食いますが仕様です(30×512×512の計算を行っているため)。

<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html>

このへんのモジュールをmodule/calc_iou.pyのconfusion matrix計算関数あたりと入れ替えたら早くなるかも

## ValidationIoU(%)

|                | mIoU | 背景 | すりガラス | 統合 | 胸水 |
|:--------------:|:----:|:----:|:----------:|:----:|:----:|
| Deeplabv3(pre) | 57.6 | 99.1 |    54.6    | 68.5 |  8.1 |
| Deeplabv3      | 59.5 | 98.4 |    67.3    | 72.2 |  0.0 |

## 結果表示コード
工事中
