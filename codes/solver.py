import torch
from torch.optim import Adam
import network
from loss import IoU_loss
import numpy as np
import cv2
from dataset import get_loader
from os.path import join
import random
from utils import mkdir, write_doc, get_time


class Solver(object):
    def __init__(self):
        self.ICNet = network.ICNet().cuda()

    def train(self, roots, init_epoch, end_epoch, learning_rate, batch_size, weight_decay, ckpt_root, doc_path, num_thread, pin, vgg_path=None):
        # 定义Adam优化器.
        optimizer = Adam(self.ICNet.parameters(),
                         lr=learning_rate, 
                         weight_decay=weight_decay)

        # 加载 ".pth" 以初始化模型.
        if init_epoch == 0:
            # 从预训练的VGG16中加载.
            self.ICNet.apply(network.weights_init)
            self.ICNet.vgg.vgg.load_state_dict(torch.load(vgg_path))
        else:
            # 从已有的检查点文件中加载.
            ckpt = torch.load(join(ckpt_root, 'Weights_{}.pth'.format(init_epoch)))
            self.ICNet.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])

        # 定义training dataloader.
        train_dataloader = get_loader(roots=roots,
                                      request=('img', 'gt', 'sism'),
                                      shuffle=True,
                                      batch_size=batch_size,
                                      data_aug=True,
                                      num_thread=num_thread,
                                      pin=pin)
        
        # 训练.
        self.ICNet.train()
        for epoch in range(init_epoch + 1, end_epoch):
            start_time = get_time()
            loss_sum = 0.0

            for data_batch in train_dataloader:
                self.ICNet.zero_grad()

                # 获得一个batch的数据.
                img, gt, sism = data_batch['img'], data_batch['gt'], data_batch['sism']
                img, gt, sism = img.cuda(), gt.cuda(), sism.cuda()

                if len(img) == 1:
                    # Batch Normalization在训练时不支持batchsize为1, 因此直接跳过该样本的训练. 
                    continue
                
                # 前向传播.
                preds_list = self.ICNet(image_group=img,
                                        SISMs=sism, 
                                        is_training=True)
                
                # 计算IoU loss.
                loss = IoU_loss(preds_list, gt)

                # 反向传播.
                loss.backward()
                optimizer.step()
                loss_sum = loss_sum + loss.detach().item()
            
            # 在每个epoch的训练后都保存检查点文件(".pth").
            mkdir(ckpt_root)
            torch.save({'optimizer': optimizer.state_dict(),
                        'state_dict': self.ICNet.state_dict()}, join(ckpt_root, 'Weights_{}.pth'.format(epoch)))
            
            # 近似地计算训练集的平均损失.
            loss_mean = loss_sum / len(train_dataloader)
            end_time = get_time()

            # 记录训练的信息到".txt"文档中.
            content = 'CkptIndex={}:    TrainLoss={}    LR={}    Time={}\n'.format(epoch, loss_mean, learning_rate, end_time - start_time)
            write_doc(doc_path, content)
    
    def test(self, roots, ckpt_path, pred_root, num_thread, batch_size, original_size, pin):
        with torch.no_grad():            
            # 加载指定的检查点文件(".pth").
            state_dict = torch.load(ckpt_path)['state_dict']
            self.ICNet.load_state_dict(state_dict)
            self.ICNet.eval()
            
            # 得到test datasets的名字.
            datasets = roots.keys()

            # 在每个dataset上对ICNet进行测试.
            for dataset in datasets:
                # 对当前dataset定义test dataloader.
                test_dataloader = get_loader(roots=roots[dataset], 
                                             request=('img', 'sism', 'file_name', 'group_name', 'size'), 
                                             shuffle=False,
                                             data_aug=False, 
                                             num_thread=num_thread, 
                                             batch_size=batch_size, 
                                             pin=pin)

                # 为当前的dataset创建文件夹以保存之后产生的预测图.
                mkdir(pred_root)
                cur_dataset_pred_root = join(pred_root, dataset)
                mkdir(cur_dataset_pred_root)

                for data_batch in test_dataloader:
                    # 获得一个batch的数据.
                    img, sism = data_batch['img'].cuda(), data_batch['sism'].cuda()

                    # 前向传播.
                    preds = self.ICNet(image_group=img, 
                                       SISMs=sism, 
                                       is_training=False)
                    
                    # 根据当前的batch所属的group创建文件夹, 以保存之后产生的预测图.
                    group_name = data_batch['group_name'][0]
                    cur_group_pred_root = join(cur_dataset_pred_root, group_name)
                    mkdir(cur_group_pred_root)

                    # preds.shape: [N, 1, H, W]->[N, H, W, 1]
                    preds = preds.permute(0, 2, 3, 1).cpu().numpy()

                    # 制作预测图的保存路径.
                    pred_paths = list(map(lambda file_name: join(cur_group_pred_root, file_name + '.png'), data_batch['file_name']))
                    
                    # 对每个预测图:
                    for i, pred_path in enumerate(pred_paths):
                        # 当 "original_size == True" 时, 将224*224大小的预测图缩放到原图尺寸.
                        H, W = data_batch['size'][0][i], data_batch['size'][1][i]
                        pred = cv2.resize(preds[i], (W, H)) if original_size else preds[i]

                        # 保存预测图.
                        cv2.imwrite(pred_path, np.array(pred * 255))