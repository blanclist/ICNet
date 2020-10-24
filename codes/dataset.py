import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import random
import numpy as np
from torch.utils import data
import PIL.ImageOps
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from os.path import join
from os import listdir

"""
build_file_paths:
    当 "file_names == None and group_names == None" 时, 
    遍历 "base" 中的文件以构建 "file_paths", "group_names", "file_names" and "indices".
    否则, 基于给定的 "file_names" 和 "group_names" 来构建 "file_paths".
"""
def build_file_paths(base, group_names=None, file_names=None, suffix='.png'):
    if file_names == None and group_names == None:
        file_paths = []
        group_names = []
        file_names = []
        indices = []
        cur_group_end_index = 0
        for group_name in listdir(base):
            group_path = join(base, group_name)
            group_file_names = listdir(group_path)
            cur_group_end_index += len(group_file_names)

            # 将当前图片组最后一张图片在整个数据集中的下标保存在 "indices" 中, 这部分信息是为 "Cosal_Sampler" 而准备的.
            indices.append(cur_group_end_index)
            
            for file_name in group_file_names:
                file_path = join(group_path, file_name)
                file_paths.append(file_path)
                group_names.append(group_name)
                file_names.append(file_name[:str(file_name).rfind('.')])
        return file_paths, group_names, file_names, indices
    else:
        file_paths = list(map(lambda i: join(base, group_names[i], file_names[i] + suffix), range(len(file_names))))
        return file_paths

"""
random_flip:
    以 0.5 的概率对输入数据进行随机水平翻转.
"""
def random_flip(img, gt, sism):
    datas = (img, gt, sism)
    if random.random() > 0.5:
        datas = tuple(map(lambda data: transforms.functional.hflip(data) if data is not None else None, datas))
    return datas


class ImageData(data.Dataset):
    def __init__(self, roots, request, aug_transform=None, rgb_transform=None, gray_transform=None):
        if 'img' in request == False:
            raise Exception('\'img\' must be contained in \'request\'.')

        self.need_gt = True if 'gt' in request else False
        self.need_file_name = True if 'file_name' in request else False
        self.need_group_name = True if 'group_name' in request else False
        self.need_sism = True if 'sism' in request else False
        self.need_size = True if 'size' in request else False

        img_paths, group_names, file_names, indices = build_file_paths(roots['img'])
        gt_paths = build_file_paths(roots['gt'], group_names, file_names) if self.need_gt else None
        sism_paths = build_file_paths(roots['sism'], group_names, file_names) if self.need_sism else None

        self.img_paths = img_paths
        self.gt_paths = gt_paths
        self.sism_paths = sism_paths
        self.file_names = file_names
        self.group_names = group_names
        self.indices = indices
        self.aug_transform = aug_transform
        self.rgb_transform = rgb_transform
        self.gray_transform = gray_transform    

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item]).convert('RGB')
        W, H = img.size
        gt = Image.open(self.gt_paths[item]).convert('L') if self.need_gt else None
        sism = Image.open(self.sism_paths[item]).convert('L') if self.need_sism else None
        group_name = self.group_names[item] if self.need_group_name else None
        file_name = self.file_names[item] if self.need_file_name else None

        if self.aug_transform is not None:
            img, gt, sism = self.aug_transform(img, gt, sism)
        
        if self.rgb_transform is not None:
            img = self.rgb_transform(img)
        if self.gray_transform is not None and self.need_gt:
            gt = self.gray_transform(gt)
        if self.gray_transform is not None and self.need_sism:
            sism = self.gray_transform(sism)
        
        data_item = {}
        data_item['img'] = img
        if self.need_gt: data_item['gt'] = gt
        if self.need_sism: data_item['sism'] = sism
        if self.need_file_name: data_item['file_name'] = file_name
        if self.need_group_name: data_item['group_name'] = group_name
        if self.need_size: data_item['size'] = (H, W)
        return data_item

    def __len__(self):
        return len(self.img_paths)


"""
Cosal_Sampler:
    提供每个batch的下标, 确保每个batch中的数据都来自于同一个图片组(属于同一个类别).
"""
class Cosal_Sampler(data.Sampler):
    def __init__(self, indices, shuffle, batch_size):
        self.indices = indices
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.len = None
        self.batches_indices = None
        self.reset_batches_indices()
    
    def reset_batches_indices(self):
        batches_indices = []
        start_idx = 0
        # 对每个图片组(属于同一个类别):
        for end_idx in self.indices:
            # 初始化 "group_indices".
            group_indices = list(range(start_idx, end_idx))
            
            # 如果 "self.shuffle == True" 则打乱 "group_indices" (打乱一组内的图片顺序).
            if self.shuffle:
                np.random.shuffle(group_indices)
            
            # 获取当前图片组的容量大小.
            num = end_idx - start_idx

            # 根据 "self.batch_size" 将 "group_indices" 划分为多个batches,
            # 然后将划分好的结果(即 "batch_indices")添加到 "batches_indices" 中.
            # 注意, 当 "self.batch_size == None" 时, 每个图片组都被直接作为一个batch (即 "batch_size = num").
            idx = 0
            while idx < num:
                batch_size = num if self.batch_size == None else self.batch_size
                batch_indices = group_indices[idx:idx + batch_size]
                batches_indices.append(batch_indices)
                idx += batch_size
            start_idx = end_idx

        # "batches_indices" 中的每个元素都是一个list, 表示某一batch的下标索引集合,
        # 但相邻的list基本上来自于同一个图片组(属于同一类别).
        # 因此, 当 "self.shuffle == True" 时, 进一步对 "batches_indices" 打乱 (打乱batch之间的顺序). 
        if self.shuffle:
            np.random.shuffle(batches_indices)
        
        self.len = len(batches_indices)
        self.batches_indices = batches_indices

    def __iter__(self):
        if self.shuffle:
            self.reset_batches_indices()
        return iter(self.batches_indices)

    def __len__(self):
        return self.len


def get_loader(roots, request, batch_size, data_aug, shuffle, num_thread=4, pin=True):
    aug_transform = random_flip if data_aug else None
    rgb_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    gray_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
        ])
    dataset = ImageData(roots, request, aug_transform=aug_transform, rgb_transform=rgb_transform, gray_transform=gray_transform)
    cosal_sampler = Cosal_Sampler(indices=dataset.indices, shuffle=shuffle, batch_size=batch_size)
    data_loader = data.DataLoader(dataset=dataset, batch_sampler=cosal_sampler, num_workers=num_thread, pin_memory=pin)
    return data_loader