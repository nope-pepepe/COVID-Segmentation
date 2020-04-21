# COVID-Segmentation
## 概要
<http://medicalsegmentation.com/covid19/> の画像をセグメンテーションするためのリポジトリ
現状とりあえずDeeplabv3+(ResNet101)で動きます
Train:70枚
Validation:30枚
で学習します。

## 学習方法
sh train.sh 
適宜書き換えてください。
BatchSizeはとりあえず4で9GBくらいメモリ食います。
メモリ並列できたら嬉しいね(将来的にやるかも)

## その他
Validation時Confusion Matrix計算で1Epochあたり10分弱食いますが仕様です(30*512*512の計算を行っているため)。

<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html>
このへんのモジュールをmodule/calc_iou.pyのconfusion matrix計算関数あたりと入れ替えたら早くなるかも

## 精度
工事中

## 結果表示コード
工事中