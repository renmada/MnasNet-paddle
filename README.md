# 模型名称 Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution
## 1. 简介
传统的图像处理,将图像按照高频和低频分离,花费更多的精力在高频部分(如图像压缩,低频数据更高的压缩率而高频数据得到更多的保留). OctConv将这种思想引入到cnn的特征图中,将特征图分离为高频部分和低频部分,同时高低频部分相互通信,卷积核也分成作用于高频和低频的部分.实验表明, OctConv 对2D和3D都有效,降低计算量的同时,精度不丢甚至有所提升

## 2. 数据集和复现精度
数据集使用ImageNet 2012的训练数据集，有1000类，训练集图片有1281167张，验证集图片有50000张，大小为144GB  
aistudio上的地址为：https://aistudio.baidu.com/aistudio/datasetdetail/79807  

|         Model        | alpha | label smoothing[2] | mixup[3] |#Params | #FLOPs |  Top1 / Top5 |
|:--------------------:|:-----:|:------------------:|:--------:|:------:|:------:|:------------:|
| 1.125 MobileNet_v2(论文)|  .5   |         Yes        |   Yes       |  4.2 M |  295 M | 73.0 / 91.2 |
| 1.125 MobileNet_v2(复现)|  .5 |         Yes        |       | 60.2 M | 10.9 G | [72.856 / -](https://dl.fbaipublicfiles.com/octconv/others/resnet152_v1f_alpha-0.125.params) |
 

### 2.1 log信息说明
训练两个阶段
1. 用config1的配置训练300epoch，对应log是log1
2. 从阶段1中val_acc最高的epoch_296开始训练，用config2的配置训练100epoch，对应log是log2


## 3. 准备环境
* 硬件：Tesla V100 * 4
* 框架：PaddlePaddle == 2.2.0
## 4. 快速开始
### 第一步：克隆本项目
```
    https://github.com/renmada/OctConv-paddle.git
```
### 第二步：放数据到dataset目录下，数据集目录
### 第三步：一阶段训练
```
    python -m paddle.distributed.launch tools/train.py -c OctMobileNetV2_1_0.yaml
```
### 第四步：二阶段训练
```
    # 需要在OctMobileNetV2_1_1.yaml文件指定epoch_296模型的路径
    python -m paddle.distributed.launch tools/train.py -c OctMobileNetV2_1_1.yaml
```

## 5. 引用
```
@article{chen2019drop,
  title={Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution},
  author={Chen, Yunpeng and Fan, Haoqi and Xu, Bing and Yan, Zhicheng and Kalantidis, Yannis and Rohrbach, Marcus and Yan, Shuicheng and Feng, Jiashi},
  journal={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
}
```