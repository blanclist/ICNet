import os
from solver import Solver

"""
训练设置(适用于 "train.py"):

vgg_path:
    预训练VGG16(".pth")的路径, 用于初始化参数来训练您自己的ICNet.

ckpt_root:
    保存检查点文件(".pth")的文件夹路径, 每个epoch训练后都会自动保存.
    第i个epoch训练完成后, 检查点文件会被保存在 "ckpt_root/Weights_{}.pth".format(i).

train_init_epoch:
    训练的起始epoch.
    当 "train_init_epoch == 0" 时, ICNet用预训练的VGG16的参数进行初始化;
    否则, ICNet加载 "ckpt_root/Weights_{}.pth".format(train_init_epoch) 处的检查点文件(".pth")来进行初始化,

train_end_epoch:
    训练的结束epoch.
    建议您训练50~60个epochs.

train_device:
    用于训练的GPU编号.

train_doc_path:
    用于保存训练过程所产生信息的文件(".txt"文档)路径.

train_roots:
    一个dict, 包含训练集图片, GTs和SISMs的文件夹路径, 其格式为:
    train_roots = {'img': 训练集的图片的文件夹路径,
                   'gt': 训练集的GTs的文件夹路径,
                   'sism': 训练集的SISMs的文件夹路径}
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

# 下面是一个构建 "train_roots" 的例子.
train_roots = {'img': '/mnt/jwd/data/COCO9213/img_bilinear_224/',
               'gt': '/mnt/jwd/data/COCO9213/gt_bilinear_224/',
               'sism': '/mnt/jwd/data/EGNet-SISMs/COCO9213/'}
# ------------ 示例结束 ------------

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