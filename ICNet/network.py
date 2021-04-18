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
    Resize tensor (shape=[N, C, H, W]) to the target size (default: 224*224).
"""
def resize(input, target_size=(224, 224)):
    return F.interpolate(input, (target_size[0], target_size[1]), mode='bilinear', align_corners=True)

"""
weights_init:
    Weights initialization.
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
    Compress the channel of input features to 1, then predict maps with sigmoid function.
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
    Two convolutional layers with residual structure.
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
    Given features extracted from the VGG16 backbone,
    exploit SISMs to build intra cues and inter cues.
"""
class Cosal_Module(nn.Module):
    def __init__(self, H, W):
        super(Cosal_Module, self).__init__()
        self.cosal_feat = Cosal_Sub_Module(H, W)
        self.conv = nn.Sequential(nn.Conv2d(256, 128, 1), Res(128))

    def forward(self, feats, SISMs):
        # Get foreground co-saliency features.
        fore_cosal_feats = self.cosal_feat(feats, SISMs)

        # Get background co-saliency features.
        back_cosal_feats = self.cosal_feat(feats, 1.0 - SISMs)
        
        # Fuse foreground and background co-saliency features
        # to generate co-saliency enhanced features.
        cosal_enhanced_feats = self.conv(torch.cat([fore_cosal_feats, back_cosal_feats], dim=1))
        return cosal_enhanced_feats

"""
Cosal_Sub_Module:
  * The kernel module of ICNet.
    Generate foreground/background co-salient features by using SISMs.
"""
class Cosal_Sub_Module(nn.Module):
    def __init__(self, H, W):
        super(Cosal_Sub_Module, self).__init__()
        channel = H * W
        self.conv = nn.Sequential(nn.Conv2d(channel, 128, 1), Res(128))

    def forward(self, feats, SISMs):
        N, C, H, W = feats.shape
        HW = H * W
        
        # Resize SISMs to the same size as the input feats.
        SISMs = resize(SISMs, [H, W])  # shape=[N, 1, H, W]
        
        # NFs: L2-normalized features.
        NFs = F.normalize(feats, dim=1)  # shape=[N, C, H, W]

        def CFM(SIVs, NFs):
            # Compute correlation maps [Figure 4] between SIVs and pixel-wise feature vectors in NFs by inner product.
            # We implement this process by ``F.conv2d()'', which takes SIVs as 1*1 kernels to convolve NFs.
            correlation_maps = F.conv2d(NFs, weight=SIVs)  # shape=[N, N, H, W]
            
            # Vectorize and normalize correlation maps.
            correlation_maps = F.normalize(correlation_maps.reshape(N, N, HW), dim=2)  # shape=[N, N, HW]
            
            # Compute the weight vectors [Equation 2].
            correlation_matrix = torch.matmul(correlation_maps, correlation_maps.permute(0, 2, 1))  # shape=[N, N, N]
            weight_vectors = correlation_matrix.sum(dim=2).softmax(dim=1)  # shape=[N, N]

            # Fuse correlation maps with the weight vectors to build co-salient attention (CSA) maps.
            CSA_maps = torch.sum(correlation_maps * weight_vectors.view(N, N, 1), dim=1)  # shape=[N, HW]
            
            # Max-min normalize CSA maps.
            min_value = torch.min(CSA_maps, dim=1, keepdim=True)[0]
            max_value = torch.max(CSA_maps, dim=1, keepdim=True)[0]
            CSA_maps = (CSA_maps - min_value) / (max_value - min_value + 1e-12)  # shape=[N, HW]
            CSA_maps = CSA_maps.view(N, 1, H, W)  # shape=[N, 1, H, W]
            return CSA_maps

        def get_SCFs(NFs):
            NFs = NFs.view(N, C, HW)  # shape=[N, C, HW]
            SCFs = torch.matmul(NFs.permute(0, 2, 1), NFs).view(N, -1, H, W)  # shape=[N, HW, H, W]
            return SCFs

        # Compute SIVs [Section 3.2, Equation 1].
        SIVs = F.normalize((NFs * SISMs).mean(dim=3).mean(dim=2), dim=1).view(N, C, 1, 1)  # shape=[N, C, 1, 1]

        # Compute co-salient attention (CSA) maps [Section 3.3].
        CSA_maps = CFM(SIVs, NFs)  # shape=[N, 1, H, W]

        # Compute self-correlation features (SCFs) [Section 3.4].
        SCFs = get_SCFs(NFs)  # shape=[N, HW, H, W]

        # Rearrange the channel order of SCFs to obtain RSCFs [Section 3.4].
        evidence = CSA_maps.view(N, HW)  # shape=[N, HW]
        indices = torch.argsort(evidence, dim=1, descending=True).view(N, HW, 1, 1).repeat(1, 1, H, W)  # shape=[N, HW, H, W]
        RSCFs = torch.gather(SCFs, dim=1, index=indices)  # shape=[N, HW, H, W]
        cosal_feat = self.conv(RSCFs * CSA_maps)  # shape=[N, 128, H, W]
        return cosal_feat

"""
Decoder_Block:
    U-net like decoder block that fuses co-saliency features and low-level features for upsampling. 
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
        # Adjust cosal_map, SISMs and old_feats to the same spatial size as low_level_feats.
        cosal_map = resize(cosal_map, [H, W])
        SISMs = resize(SISMs, [H, W])
        old_feats = resize(old_feats, [H, W])

        # Predict co-saliency maps with the size of H*W.
        cmprs = self.cmprs(low_level_feats)
        new_feats = self.merge_conv(torch.cat([cmprs * cosal_map, 
                                               cmprs * SISMs, 
                                               old_feats], dim=1))
        new_cosal_map = self.pred(new_feats)
        return new_feats, new_cosal_map


"""
ICNet:
    The entire ICNet.
    Given a group of images and corresponding SISMs, ICNet outputs a group of co-saliency maps (predictions) at once.
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
        # Extract features from the VGG16 backbone.
        conv1_2 = self.vgg(image_group, 'conv1_1', 'conv1_2_mp') # shape=[N, 64, 224, 224]
        conv2_2 = self.vgg(conv1_2, 'conv1_2_mp', 'conv2_2_mp')  # shape=[N, 128, 112, 112]
        conv3_3 = self.vgg(conv2_2, 'conv2_2_mp', 'conv3_3_mp')  # shape=[N, 256, 56, 56]
        conv4_3 = self.vgg(conv3_3, 'conv3_3_mp', 'conv4_3_mp')  # shape=[N, 512, 28, 28]
        conv5_3 = self.vgg(conv4_3, 'conv4_3_mp', 'conv5_3_mp')  # shape=[N, 512, 14, 14]

        # Compress the channels of high-level features.
        conv6_cmprs = self.conv6_cmprs(conv5_3)  # shape=[N, 128, 7, 7]
        conv5_cmprs = self.conv5_cmprs(conv5_3)  # shape=[N, 256, 14, 14]
        conv4_cmprs = self.conv4_cmprs(conv4_3)  # shape=[N, 256, 28, 28]

        # Obtain co-saliancy features.
        cosal_feat_6 = self.Co6(conv6_cmprs, SISMs) # shape=[N, 128, 7, 7]
        cosal_feat_5 = self.Co5(conv5_cmprs, SISMs) # shape=[N, 128, 14, 14]
        cosal_feat_4 = self.Co4(conv4_cmprs, SISMs) # shape=[N, 128, 28, 28]
        
        # Merge co-saliancy features and predict co-saliency maps with size of 28*28 (i.e., "cosal_map_4").
        feat_56 = self.merge_co_56(cosal_feat_5 + resize(cosal_feat_6, [14, 14])) # shape=[N, 128, 14, 14]
        feat_45 = self.merge_co_45(cosal_feat_4 + resize(feat_56, [28, 28]))      # shape=[N, 128, 28, 28]
        cosal_map_4 = self.get_pred_4(feat_45)                                    # shape=[N, 1, 28, 28]

        # Obtain co-saliency maps with size of 224*224 (i.e., "cosal_map_1") by progressively upsampling.
        feat_34, cosal_map_3 = self.refine_3(conv3_3, cosal_map_4, SISMs, feat_45)
        feat_23, cosal_map_2 = self.refine_2(conv2_2, cosal_map_4, SISMs, feat_34)
        _, cosal_map_1 = self.refine_1(conv1_2, cosal_map_4, SISMs, feat_23)      # shape=[N, 1, 224, 224]

        # Return predicted co-saliency maps.
        if is_training:
            preds_list = [resize(cosal_map_4), resize(cosal_map_3), resize(cosal_map_2), cosal_map_1]
            return preds_list
        else:
            preds = cosal_map_1
            return preds
