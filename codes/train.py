import os
from solver import Solver, set_seed

"""
Training settings (used for "train.py"):

vgg_path:
    The path of pre-trained VGG16 (".pth") used to initialize model at the start of training.

ckpt_root:
    The path of folder used to save the checkpoint file (".pth").

train_init_epoch:
    Starting epoch of the training.
    When "train_init_epoch == 0", the model is initialized with pre-trained VGG16;
    Otherwise, the model loads corresponding checkpoint file (from "ckpt_root") for initialization.

train_end_epoch:
    Ending epoch of the training.
    We recommond you to train ICNet for 50~60 epochs.

train_device:
    Index of GPU used for training.

train_doc_path:
    The path of file (".txt") used to save the training information.

train_roots:
    A dictionary containing paths of the training dataset.
    'img': Images folder path.
    'gt': Ground-truths folder path.
    'sism': SISMs folder path.
    (SISMs: Single Image Saliency Maps produced by any off-the-shelf SOD model.)
"""

vgg_path = './vgg16_feat.pth'
ckpt_root = './ckpt/'
train_init_epoch = 0
train_end_epoch = 61
train_device = '0'
train_doc_path = './training.txt'
learning_rate = 1e-5
weight_decay = 1e-4
train_batch_size = 10
train_num_thread = 4
train_roots = {'img': '/mnt/jwd/data/COCO9213/img_bilinear_224/',
               'gt': '/mnt/jwd/data/COCO9213/gt_bilinear_224/',
               'sism': '/mnt/jwd/data/EGNet-SISMs/COCO9213/'}

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = train_device
    solver = Solver()
    solver.train(roots=train_roots,
                 vgg_path=vgg_path,
                 init_epoch=train_init_epoch,
                 end_epoch=train_end_epoch,
                 learning_rate=learning_rate,
                 batch_size=train_batch_size,
                 weight_decay=weight_decay,
                 ckpt_root=ckpt_root,
                 doc_path=train_doc_path,
                 num_thread=train_num_thread,
                 pin=False)