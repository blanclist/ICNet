import os
from solver import Solver

"""
Test settings (used for "test.py"):

test_device:
    Index of GPU used for test.

test_batch_size:
    Test batchsize.
  * When "test_batch_size == None", the dataloader takes the whole image group as a batch to
    perform test (regardless of the size of image group). If your GPU does not have enough memory,
    you are suggested to set "test_batch_size" with a specific small number (e.g. test_batch_size = 10).

pred_root:
    The path of folder used to save predictions (co-saliency maps).

ckpt_path:
    The path of checkpoint file (".pth") used for test.

original_size:
    When "original_size == True", the prediction of ICNet will be resized to the size of input image.

test_roots:
    A dictionary including multiple sub-dictionary, each of them contains paths of the test dataset.
    'img': Images folder path.
    'sism': SISMs folder path.
    (SISMs: Single Image Saliency Maps produced by any off-the-shelf SOD model.)
"""

test_device = '0'
test_batch_size = None
pred_root = './pred/'
ckpt_path = './ICNet_vgg16.pth'
original_size = False
test_num_thread = 4

# An example to build "test_roots".
test_roots = dict()
datasets = ['MSRC', 'iCoSeg', 'CoSal2015', 'CoSOD3k', 'CoCA']

for dataset in datasets:
    roots = {'img': '/mnt/jwd/data/{}/img_bilinear_224/'.format(dataset),
             'sism': '/mnt/jwd/data/EGNet-SISMs/{}/'.format(dataset)}
    test_roots[dataset] = roots

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