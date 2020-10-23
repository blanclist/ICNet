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
    When "file_names == None and group_names == None", 
    traverse file folder to build "file_paths", "group_names", "file_names" and "indices".
    Otherwise, build "file_paths" based on given "file_names" and "group_names".
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

            # Save the ending index of current group into "indices", which is prepared for "Cosal_Sampler".
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
    Flip inputs horizontally with a possibility of 0.5.
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
    Provide indices of each batch, ensuring that each batch data is extracted from the same image group (with the same category).
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
        # For each image group (with same category):
        for end_idx in self.indices:
            # Initalize "group_indices".
            group_indices = list(range(start_idx, end_idx))
            
            # Shuffle "group_indices" if needed.
            if self.shuffle:
                np.random.shuffle(group_indices)
            
            # Get the size of current image group.
            num = end_idx - start_idx

            # Split "group_indices" to multiple batches according to "self.batch_size",
            # then append the splited indices ("batch_indices") to "batches_indices".
            # Note that, when "self.batch_size == None", each image group is regarded as a batch ("batch_size = num").
            idx = 0
            while idx < num:
                batch_size = num if self.batch_size == None else self.batch_size
                batch_indices = group_indices[idx:idx + batch_size]
                batches_indices.append(batch_indices)
                idx += batch_size
            start_idx = end_idx

        # Each entry of "batches_indices" is a list indicating indices of a specific batch,
        # but neighbouring entries basically belongs to the same image group (with same category).
        # Thus, shuffle "batches_indices" if needed. 
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