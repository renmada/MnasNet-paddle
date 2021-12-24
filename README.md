# 基于PaddleClas复现 Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution
## 1. 简介
传统的图像处理,将图像按照高频和低频分离,花费更多的精力在高频部分(如图像压缩,低频数据更高的压缩率而高频数据得到更多的保留). OctConv将这种思想引入到cnn的特征图中,将特征图分离为高频部分和低频部分,同时高低频部分相互通信,卷积核也分成作用于高频和低频的部分.实验表明, OctConv 对2D和3D都有效,降低计算量的同时,精度不丢甚至有所提升

## 2. 数据集和复现精度
数据集使用ImageNet 2012的训练数据集，有1000类，训练集图片有1281167张，验证集图片有50000张，大小为144GB  
aistudio上的地址为：https://aistudio.baidu.com/aistudio/datasetdetail/79807  

|         Model        | alpha | label smoothing[2] | mixup[3] |#Params | #FLOPs |  Top1 / Top5 |
|:--------------------:|:-----:|:------------------:|:--------:|:------:|:------:|:------------:|
| 1.125 MobileNet_v2(论文)|  .5   |         Yes        |   Yes       |  4.2 M |  295 M | 73.0 / 91.2 |
| 1.125 MobileNet_v2(复现)|  .5 |         Yes        |   Yes    | 4.2 M | - |  72.95 / - |
 

### 2.1 log信息说明
训练分为两个阶段
1. 用[config1](train1.yaml)的配置训练200epoch
2. 加载阶段1的epoch_200权重，用[config2](train2.yaml)的配置训练200epoch


## 3. 准备环境
* 硬件：Tesla V100 * 4
* 框架：PaddlePaddle == 2.2.0
## 4. 快速开始
### 4.1克隆本项目
```
https://github.com/renmada/OctConv-paddle.git
```
### 4.2 下载数据集，放到指定位置
### 4.3 一阶段训练
```
# 修改train1.yaml中的output_dir image_root cls_label_path的路径

python -m paddle.distributed.launch tools/train.py -c train1.yaml
```
### 4.4 二阶段训练
```
# 修改train2.yaml中的output_dir image_root cls_label_path pretrained_model的路径

python -m paddle.distributed.launch tools/train.py -c train2.yaml
```
此阶段可以在[aistudio](https://aistudio.baidu.com/aistudio/clusterprojectdetail/3199634)上直接运行

### 4.5 相关文件
|         阶段        | log | 权重 |
|:--------------------:|:-----:|:------------------:|
| stage1|  [stage1.log](./log/stage1.log)   | [epoch_200](https://aistudio.baidu.com/aistudio/datasetdetail/122215)|  
| stage2|  [stage2.log](./log/stage2.log)|  [best_model](https://aistudio.baidu.com/aistudio/datasetdetail/122215) | 
|eval|[eval.log](./log/eval.log)|[best_model](https://aistudio.baidu.com/aistudio/datasetdetail/122215)|

**[模型网络代码](./ppcls/arch/backbone/model_zoo/oct_mobilenet_v2.py)**

### 4.6 评估
```
# 修改eval.yaml中的output_dir image_root cls_label_path pretrained_model的路径

python  tools/eval.py -c eval.yaml
```
### 4.7 预测
```
# infer.yaml中的output_dir image_root cls_label_path pretrained_model的路径

python tools/infer.py -c infer.yaml
```

demo图片预测结果
```
[{'class_ids': [285, 281, 282, 287, 286], 'scores': [0.27909, 0.20617, 0.06177, 0.01999, 0.0168], 'file_name': 'demo/cat.jpg', 'label_names': ['Egyptian cat', 'tabby, tabby cat', 'tiger cat', 'lynx, catamount', 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor']}]
```

## 5 引用
```
@article{chen2019drop,
  title={Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution},
  author={Chen, Yunpeng and Fan, Haoqi and Xu, Bing and Yan, Zhicheng and Kalantidis, Yannis and Rohrbach, Marcus and Yan, Shuicheng and Feng, Jiashi},
  journal={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
}
```