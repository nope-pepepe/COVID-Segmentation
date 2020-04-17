#python3 main.py -g 0 -e 100 -b 32 -lr 0.1 -s 1 --scheduler -m VGG16 --modelname VGG_1.pth
#python3 main.py -g 0 -e 100 -b 32 -lr 0.1 -s 2 --scheduler -m VGG16 --modelname VGG_2.pth
#python3 main.py -g 0 -e 100 -b 32 -lr 0.1 -s 3 --scheduler -m VGG16 --modelname VGG_3.pth

python3 main.py -g 1 -e 100 -b 32 -lr 0.1 -s 1 --scheduler -m ResNet101 --modelname Res_1.pth
python3 main.py -g 1 -e 100 -b 32 -lr 0.1 -s 2 --scheduler -m ResNet101 --modelname Res_2.pth
python3 main.py -g 1 -e 100 -b 32 -lr 0.1 -s 3 --scheduler -m ResNet101 --modelname Res_3.pth