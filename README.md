# **ICNet: Intra-saliency Correlation Network for Co-Saliency Detection**

本页提供了我们在 *NeurIPS(2020)* 所发表论文的 PyTorch 官方实现。

您可以通过切换 branch 到 "ICNet" 来查看英文界面。

(You can view this page in English by switching the branch to "ICNet")

<div align=center><img width="450" height="300" src=./thumbnail.png/></div>

## 训练集

**我们的训练集是 *COCO* 数据集的子集, 包含 9213 张图片.**

* ***COCO9213-os.zip*** (原始图片大小, 4.53GB), [GoogleDrive](https://drive.google.com/file/d/1fOfSX_CtWizDapB0OeTJxAydL2yDOP5H/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1wOxdP6EQEqMwjg3_v1z2-A) (提取码: 5183)。

* ***COCO9213.zip*** (放缩至224*224, 943MB), [GoogleDrive](https://drive.google.com/file/d/1GbA_WKvJm04Z1tR8pTSzBdYVQ75avg4f/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1r-qCLeG3L6i-OrBfKrXANg) (提取码: 8d7z)。

## 测试集

### 在论文中使用的数据集:

* ***MSRC*** (7 组, 233 张图片) ''Object Categorization by Learned Universal Visual Dictionary, *ICCV(2005)*''

* ***iCoseg*** (38 组, 643 张图片) ''iCoseg: Interactive Co-segmentation with Intelligent Scribble Guidance, *CVPR(2010)*''

* ***Cosal2015*** (50 组, 2015 张图片) Detection of Co-salient Objects by Looking Deep and Wide, *IJCV(2016)*''

您可以从下面的链接进行下载:

***test-datasets*** (放缩至224*224, 77MB) [GoogleDrive](https://drive.google.com/drive/folders/1bjI2msek72dOejmK796tXyjFPIE27267?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1KX7m0g9mgACoTMgkbIjRvw) (提取码: oq5w)。

***test-datasets-os*** (原始图片大小, 142MB) [GoogleDrive](https://drive.google.com/drive/folders/1p--uTLIF-2hRIJk9Xmys9ftTdXrWYslS?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1kDv7icEDT5pPwQQJkHkgpA) (提取码: ujdl)。

### 最近发布的数据集:

* **[*CoSOD3k*](http://dpfan.net/CoSOD3K/)** (160 组, 3316 张图片) ''Taking a Deeper Look at the Co-salient Object Detection, *CVPR(2020)*''

* **[*CoCA*](http://zhaozhang.net/coca.html)** (80 组, 1295 张图片) ''Gradient-Induced Co-Saliency Detection, *ECCV(2020)*''

## 预训练模型

我们提供了预训练的 ICNet ，其使用的 SISMs 是由预训练的 [EGNet](https://github.com/JXingZhao/EGNet) (基于VGG16) 产生的。

***ICNet_vgg16.pth*** (70MB), [GoogleDrive](https://drive.google.com/file/d/1wcT_XmwlshbLqCiJetmzQwi1ZNAzxiSU/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1__iiBcAI2S-Ns9MZnZwp8g) (提取码: nkj9)。

## 预测结果

我们提供了 ICNet 在 5 个基准数据集上产生的 co-saliency maps (预测结果)， 包括：

***MSRC***, ***iCoseg***, ***Cosal2015***, ***CoCA***, 和 ***CoSOD3k***。

***cosal-maps.zip*** (224*224分辨率，由模型直接输出, 20MB), [GoogleDrive](https://drive.google.com/file/d/1q9CAzPf5U3VPa_DGxzUGI_DANCuw_WEk/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1qbPJKMTiVStqjSGYWuqSgQ) (提取码: du5e)。

***cosal-maps-os.zip*** (放缩至原始图片大小, 62MB), [GoogleDrive](https://drive.google.com/file/d/1px4tPVWAgbBPMt6Rp23oNwWz8Ulj6pmX/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1WFQxeIOjOiByiFYHLpuytA) (提取码: xwcv)。

## 训练和测试

### 准备 SISMs

ICNet 可以基于任意现成 SOD 方法产生的 SISMs 进行训练和测试，但我们建议您在训练和测试阶段使用**相同**的 SOD 方法来生成 SISMs ，以确保在训练和测试时的一致性。

在论文中，我们选择预训练的 [EGNet](https://github.com/JXingZhao/EGNet) (基于VGG16) 作为基础 SOD 模型来产生 SISMs。您可以直接从下面的链接下载这些已经生成好的 SISMs：

***EGNet-SISMs*** (放缩至224*224, 125MB) [GoogleDrive](https://drive.google.com/drive/folders/1cGtXQI2U8pH37-mgSw3otnMsRi36QwBp?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/11xJz-_TPXaL0cnwUYFUOsw) (提取码: xc5k)。

### 训练

1. 下载预训练的 VGG16 ：

   ***vgg16_feat.pth*** (56MB) [GoogleDrive](https://drive.google.com/file/d/1ej5ngj2NYH-R-0GfYUDfuM-DNLuFolED/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1S_D6qCE2vn_okBhT1Zg72g) (提取码: imsf)。

2. 根据 **"./ICNet/train.py"** 中的说明修改训练设置。

3. 运行：

```
python ./ICNet/train.py
```

### 测试

1. * 测试**预训练的** ICNet：

     下载预训练的 ICNet ***"ICNet_vgg16.pth"*** (下载链接已在先前给出)。

   * 测试**您自己训练的** ICNet：

     选择您想要加载的检查点文件 ***"Weights_i.pth"***  (在第 i 个 epoch 训练后会自动保存)。

2. 根据 **"./ICNet/test.py"** 中的说明修改测试设置。

3. 运行：

```
python ./ICNet/test.py
```

## 评测

文件夹 "./ICNet/evaluator/" 包含了用 PyTorch (GPU版本) 实现的评测代码, 评测指标有 **max F-measure**, **S-measure** 以及 **MAE** 。

1. 根据 **"./ICNet/evaluate.py"** 中的说明修改评测设置。

2. 运行：

```
python ./ICNet/evaluate.py
```

## 比较的方法

我们将 ICNet 与 7 个 state-of-the-art Co-SOD 方法进行了比较：

* ***CBCS***		''Cluster-Based Co-Saliency Detection, *TIP(2013)*''​			  

* ***CSHS***		''Co-Saliency Detection Based on Hierarchical Segmentation, *SPL(2014)*''

* ***CoDW***		''Detection of Co-salient Objects by Looking Deep and Wide, *IJCV(2016)*''

* ***UCSG***		''Unsupervised CNN-based Co-Saliency Detection with Graphical Optimization, *ECCV(2018)*''

* ***CSMG***		''Co-saliency Detection via Mask-guided Fully Convolutional Networks with Multi-scale Label Smoothing, *CVPR(2019)*''

* ***MGLCN***		''A Unified Multiple Graph Learning and Convolutional Network Model for Co-saliency Estimation, *ACM MM(2019)*''

* ***GICD***		''Gradient-Induced Co-Saliency Detection, *ECCV(2020)*''

您可以从下面的链接下载这些方法产生的预测结果图：

***compared_method*s** (原始图片大小, 445MB) [GoogleDrive](https://drive.google.com/drive/folders/1qdXWZQ-fF-WaCF-rat0Da7vFrAIYsj09?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/10vpubz39atkg2lz095QvSQ) (提取码: s7pr)。

## 引用

*待更新。*

## 联系方式

如果您有任何问题，可以随时通过 jwd331@126.com 与我 (金闻达) 联系，我会尽快回复。