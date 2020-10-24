import torch
import time
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from os.path import join
np.set_printoptions(suppress=True, threshold=1e5)

"""
resize:
    将tensor (shape=[N, C, H, W]) 双线性放缩到 "target_size" 大小 (默认: 224*224).
"""
def resize(input, target_size=(224, 224)):
    return F.interpolate(input, (target_size[0], target_size[1]), mode='bilinear', align_corners=True)

"""
weights_init:
    权重初始化.
"""
def weights_init(module):
    if isinstance(module, nn.Conv2d):
        init.normal_(module.weight, 0, 0.01)
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)


""""
VGG16:
    VGG16 backbone.
""" 
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        layers = []
        in_channel = 3
        vgg_out_channels = (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M')
        for out_channel in vgg_out_channels:
            if out_channel == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channel = out_channel
        self.vgg = nn.ModuleList(layers)
        self.table = {'conv1_1': 0, 'conv1_2': 2, 'conv1_2_mp': 4,
                      'conv2_1': 5, 'conv2_2': 7, 'conv2_2_mp': 9,
                      'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14, 'conv3_3_mp': 16,
                      'conv4_1': 17, 'conv4_2': 19, 'conv4_3': 21, 'conv4_3_mp': 23,
                      'conv5_1': 24, 'conv5_2': 26, 'conv5_3': 28, 'conv5_3_mp': 30, 'final': 31}

    def forward(self, feats, start_layer_name, end_layer_name):
        start_idx = self.table[start_layer_name]
        end_idx = self.table[end_layer_name]
        for idx in range(start_idx, end_idx):
            feats = self.vgg[idx](feats)
        return feats


"""
Prediction:
    将输入特征的通道压缩到1维, 然后利用sigmoid函数产生预测图.
"""
class Prediction(nn.Module):
    def __init__(self, in_channel):
        super(Prediction, self).__init__()
        self.pred = nn.Sequential(nn.Conv2d(in_channel, 1, 1), nn.Sigmoid())

    def forward(self, feats):
        pred = self.pred(feats)
        return pred


"""
Res:
    带有残差结构的卷积层.
"""
class Res(nn.Module):
    def __init__(self, in_channel):
        super(Res, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 1), 
                                  nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channel, in_channel, 3, 1, 1))

    def forward(self, feats):
        feats = feats + self.conv(feats)
        feats = F.relu(feats, inplace=True)
        return feats

"""
Cosal_Module:
    给定从VGG16中抽取出的特征，
    利用SISMs构建intra cues和inter cues.
"""
class Cosal_Module(nn.Module):
    def __init__(self, H, W):
        super(Cosal_Module, self).__init__()
        self.cosal_feat = Cosal_Sub_Module(H, W)
        self.conv = nn.Sequential(nn.Conv2d(256, 128, 1), Res(128))

    def forward(self, feats, SISMs):
        # 获取foreground co-saliency features.
        fore_cosal_feats = self.cosal_feat(feats, SISMs)

        # 获取background co-saliency features.
        back_cosal_feats = self.cosal_feat(feats, 1.0 - SISMs)
        
        # 融合 "fore_cosal_feats" 和 "fore_cosal_feats",
        # 产生co-saliency enhanced features.
        cosal_enhanced_feats = self.conv(torch.cat([fore_cosal_feats, back_cosal_feats], dim=1))
        return cosal_enhanced_feats

"""
Cosal_Sub_Module:
  * ICNet的核心单元.
    利用SISMs产生foreground/background co-salient features.
"""
class Cosal_Sub_Module(nn.Module):
    def __init__(self, H, W):
        super(Cosal_Sub_Module, self).__init__()
        channel = H * W
        self.conv = nn.Sequential(nn.Conv2d(channel, 128, 1), Res(128))

    def forward(self, feats, SISMs):
        N, C, H, W = feats.shape
        HW = H * W
        
        # 将SISMs调整到和输入特征feats一样的尺度.
        SISMs = resize(SISMs, [H, W])  # shape=[N, 1, H, W]
        
        # NFs: L2标准化(normalize)后的特征.
        NFs = F.normalize(feats, dim=1)  # shape=[N, C, H, W]

        def CFM(SIVs, NFs):
            # 计算SIVs和NFs中每个像素的内积, 产生correlation maps [图4].
            # 我们通过 ``F.conv2d()'' 来实现这一过程, 其中将SIVs作为1*1卷积的参数对NFs进行卷积.
            NFs = NFs.permute(1, 2, 3, 0).reshape(1, C, HW, N)  # shape=[1, C, HW, N]
            correlation_maps = F.conv2d(NFs, weight=SIVs).permute(3, 1, 2, 0)  # shape=[N, N, HW, 1]
            
            # 向量化(vectorize)并标准化(normalize) correlation maps.
            correlation_maps = F.normalize(correlation_maps.reshape(N, N, HW), dim=2)  # shape=[N, N, HW]
            
            # 计算权重向量(weight vectors) [式2].
            correlation_matrix = torch.matmul(correlation_maps, correlation_maps.permute(0, 2, 1))  # shape=[N, N, N]
            weight_vectors = correlation_matrix.sum(dim=2).softmax(dim=1)  # shape=[N, N]

            # 根据权重向量(weight vectors)对correlation maps进行融合, 产生co-salient attention (CSA) maps.
            CSA_maps = torch.sum(correlation_maps * weight_vectors.view(N, N, 1), dim=1)  # shape=[N, HW]
            
            # Max-min normalize CSA maps (将CSA maps的范围拉伸至0~1之间).
            min_value = torch.min(CSA_maps, dim=1, keepdim=True)[0]
            max_value = torch.max(CSA_maps, dim=1, keepdim=True)[0]
            CSA_maps = (CSA_maps - min_value) / (max_value - min_value + 1e-12)  # shape=[N, HW]
            CSA_maps = CSA_maps.view(N, 1, H, W)  # shape=[N, 1, H, W]
            return CSA_maps

        def get_SCFs(NFs):
            NFs = NFs.view(N, C, HW)  # shape=[N, C, HW]
            SCFs = torch.matmul(NFs.permute(0, 2, 1), NFs).view(N, -1, H, W)  # shape=[N, HW, H, W]
            return SCFs

        # 计算 SIVs [3.2节, 式1].
        SIVs = F.normalize((NFs * SISMs).mean(dim=3).mean(dim=2), dim=1).view(N, C, 1, 1)  # shape=[N, C, 1, 1]

        # 计算 co-salient attention (CSA) maps [3.3节].
        CSA_maps = CFM(SIVs, NFs)  # shape=[N, 1, H, W]

        # 计算 self-correlation features (SCFs) [3.4节].
        SCFs = get_SCFs(NFs)  # shape=[N, HW, H, W]

        # 重排列(Rearrange)SCFs的通道顺序, 产生RSCFs [3.4节].
        evidence = CSA_maps.view(N, HW)  # shape=[N, HW]
        indices = torch.argsort(evidence, dim=1, descending=True).view(N, HW, 1, 1).repeat(1, 1, H, W)  # shape=[N, HW, H, W]
        RSCFs = torch.gather(SCFs, dim=1, index=indices)  # shape=[N, HW, H, W]
        cosal_feat = self.conv(RSCFs * CSA_maps)  # shape=[N, 128, H, W]
        return cosal_feat

"""
Refinement:
    U-net风格的decoder block, 融合co-saliency features和low-level features以进行上采样.
"""
class Decoder_Block(nn.Module):
    def __init__(self, in_channel):
        super(Decoder_Block, self).__init__()
        self.cmprs = nn.Conv2d(in_channel, 32, 1)
        self.merge_conv = nn.Sequential(nn.Conv2d(96, 96, 3, 1, 1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
                                        nn.Conv2d(96, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.pred = Prediction(32)

    def forward(self, low_level_feats, cosal_map, SISMs, old_feats):
        _, _, H, W = low_level_feats.shape
        # 调整cosal_map, SISMs和old_feats的大小, 使其与low_level_feats的大小一致.
        cosal_map = resize(cosal_map, [H, W])
        SISMs = resize(SISMs, [H, W])
        old_feats = resize(old_feats, [H, W])

        # 预测大小为H*W的co-saliency maps.
        cmprs = self.cmprs(low_level_feats)
        new_feats = self.merge_conv(torch.cat([cmprs * cosal_map, 
                                               cmprs * SISMs, 
                                               old_feats], dim=1))
        new_cosal_map = self.pred(new_feats)
        return new_feats, new_cosal_map


"""
ICNet:
    整体的ICNet.
    对给定的一组图片和对应的SISMs, ICNet一次性输出这一组的co-saliency maps(预测图).
"""
class ICNet(nn.Module):
    def __init__(self):
        super(ICNet, self).__init__()
        self.vgg = VGG16()
        self.Co6 = Cosal_Module(7, 7)
        self.Co5 = Cosal_Module(14, 14)
        self.Co4 = Cosal_Module(28, 28)
        self.conv6_cmprs = nn.Sequential(nn.MaxPool2d(2, 2), nn.Conv2d(512, 128, 1),
                                         nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                         nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                         nn.Conv2d(128, 128, 3, 1, 1))
        self.conv5_cmprs = nn.Conv2d(512, 256, 1)
        self.conv4_cmprs = nn.Conv2d(512, 256, 1)

        self.merge_co_56 = Res(128)
        self.merge_co_45 = nn.Sequential(Res(128), nn.Conv2d(128, 32, 1))
        self.get_pred_4 = Prediction(32)
        self.refine_3 = Decoder_Block(256)
        self.refine_2 = Decoder_Block(128)
        self.refine_1 = Decoder_Block(64)

    def forward(self, image_group, SISMs, is_training):
        # 从VGG16 backbone中提取特征.
        conv1_2 = self.vgg(image_group, 'conv1_1', 'conv1_2_mp') # shape=[N, 64, 224, 224]
        conv2_2 = self.vgg(conv1_2, 'conv1_2_mp', 'conv2_2_mp')  # shape=[N, 128, 112, 112]
        conv3_3 = self.vgg(conv2_2, 'conv2_2_mp', 'conv3_3_mp')  # shape=[N, 256, 56, 56]
        conv4_3 = self.vgg(conv3_3, 'conv3_3_mp', 'conv4_3_mp')  # shape=[N, 512, 28, 28]
        conv5_3 = self.vgg(conv4_3, 'conv4_3_mp', 'conv5_3_mp')  # shape=[N, 512, 14, 14]

        # 对high-level features的特征先进行一次压缩.
        conv6_cmprs = self.conv6_cmprs(conv5_3)  # shape=[N, 128, 7, 7]
        conv5_cmprs = self.conv5_cmprs(conv5_3)  # shape=[N, 256, 14, 14]
        conv4_cmprs = self.conv4_cmprs(conv4_3)  # shape=[N, 256, 28, 28]

        # 获得co-saliancy features.
        cosal_feat_6 = self.Co6(conv6_cmprs, SISMs) # shape=[N, 128, 7, 7]
        cosal_feat_5 = self.Co5(conv5_cmprs, SISMs) # shape=[N, 128, 14, 14]
        cosal_feat_4 = self.Co4(conv4_cmprs, SISMs) # shape=[N, 128, 28, 28]
        
        # 融合co-saliancy features并预测尺度为28*28的co-saliency maps(即, "cosal_map_4").
        feat_56 = self.merge_co_56(cosal_feat_5 + resize(cosal_feat_6, [14, 14])) # shape=[N, 128, 14, 14]
        feat_45 = self.merge_co_45(cosal_feat_4 + resize(feat_56, [28, 28]))      # shape=[N, 128, 28, 28]
        cosal_map_4 = self.get_pred_4(feat_45)                                    # shape=[N, 1, 28, 28]

        # 通过逐渐上采样来获得尺度为224*224的co-saliency maps(即, "cosal_map_1").
        feat_34, cosal_map_3 = self.refine_3(conv3_3, cosal_map_4, SISMs, feat_45)
        feat_23, cosal_map_2 = self.refine_2(conv2_2, cosal_map_4, SISMs, feat_34)
        _, cosal_map_1 = self.refine_1(conv1_2, cosal_map_4, SISMs, feat_23)      # shape=[N, 1, 224, 224]

        # 返回预测的co-saliency maps.
        if is_training:
            preds_list = [resize(cosal_map_4), resize(cosal_map_3), resize(cosal_map_2), cosal_map_1]
            return preds_list
        else:
            preds = cosal_map_1
            return preds