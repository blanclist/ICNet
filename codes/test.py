import os
from solver import Solver

"""
测试设置(适用于 "test.py"):

test_device:
    用于测试的GPU编号.

test_batch_size:
    测试时的batchsize.
  * 当 "test_batch_size == None" 时, dataloader直接将整个图片组作为一个batch来进行测试(不论这个图片组包含多少张图片).
    如果您的GPU没有足够的显存, 您可以为 "test_batch_size" 指定一个较小的值(例如 test_batch_size = 10).

pred_root:
    用于保存预测图(co-saliency maps)的文件夹路径.

ckpt_path:
    待测试的".pth"文件的路径.

original_size:
    当 "original_size == True" 时, ICNet产生的预测图(224*224)会被缩放至原图尺寸后再保存.

test_roots:
    一个包含多个子dict的dict, 其中每个子dict应包含某个数据集的图片和对应SISMs的文件夹路径, 其格式为:
    test_roots = {
        数据集1的名称: {
            'img': 数据集1的图片的文件夹路径,
            'sism': 数据集1的SISMs的文件夹路径
        },
        数据集2的名称: {
            'img': 数据集2的图片的文件夹路径,
            'sism': 数据集2的SISMs的文件夹路径
        }
        .
        .
        .
    }
"""

test_device = '0'
test_batch_size = None
pred_root = './pred/'
ckpt_path = './ICNet_vgg16.pth'
original_size = False
test_num_thread = 4

# 下面是一个构建 "test_roots" 的例子.
test_roots = dict()
datasets = ['MSRC', 'iCoSeg', 'CoSal2015', 'CoSOD3k', 'CoCA']

for dataset in datasets:
    roots = {'img': '/mnt/jwd/data/{}/img_bilinear_224/'.format(dataset),
             'sism': '/mnt/jwd/data/EGNet-SISMs/{}/'.format(dataset)}
    test_roots[dataset] = roots
# ------------ 示例结束 ------------

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = test_device
    solver = Solver()
    solver.test(roots=test_roots,
                ckpt_path=ckpt_path,
                pred_root=pred_root, 
                num_thread=test_num_thread, 
                batch_size=test_batch_size, 
                original_size=original_size,
                pin=False)