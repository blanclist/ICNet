import os
from evaluator.evaluator import evaluate_dataset
from utils import write_doc

"""
* 注意:
    为了加速运算, "./evaluator/" 中的评测代码是通过GPU版本的PyTorch实现的。

    由于一些GTs(例如在 "Cosal2015" 测试集中)的原始分辨率很大，以致于无法通过GPU对其进行并行的指标评测
    (我们使用的 "Titan Xp" 有12G显存, 但在计算F-measure时仍然不够), 因此我们的评测代码会在计算指标前，
    将输入的预测图和对应的GTs都放缩至224*224的分辨率.
"""

"""
evaluate:
    对给定的预测图, 计算多个评测指标(max F-measure, S-measure and MAE).
    评测的结果被保存在 "doc_path" 中.
"""
def evaluate(roots, doc_path, num_thread, pin):
    datasets = roots.keys()
    for dataset in datasets:
        # 对 "dataset" 的预测图进行评测.
        results = evaluate_dataset(roots=roots[dataset], 
                                   dataset=dataset,
                                   batch_size=1, 
                                   num_thread=num_thread, 
                                   demical=True,
                                   suffixes={'gt': '.png', 'pred': '.png'},
                                   pin=pin)
        
        # 保存评测结果.
        content = '{}:\n'.format(dataset)
        content += 'max-Fmeasure={}'.format(results['max_f'])
        content += ' '
        content += 'Smeasure={}'.format(results['s'])
        content += ' '
        content += 'MAE={}\n'.format(results['mae'])
        write_doc(doc_path, content)
    content = '\n'
    write_doc(doc_path, content)

"""
评测设置(适用于 "evaluate.py"):

eval_device:
    用于评测的GPU编号.

eval_doc_path:
    用于保存评测结果的".txt"文档路径.

eval_roots:
    一个包含多个子dict的dict, 其中每个子dict应包含某个数据集的预测图和对应GTs的文件夹路径, 其格式为:
    eval_roots = {
        数据集1的名称: {
            'gt': 数据集1的GTs的文件夹路径,
            'pred': 数据集1的预测图的文件夹路径
        },
        数据集2的名称: {
            'gt': 数据集2的GTs的文件夹路径,
            'pred': 数据集2的预测图的文件夹路径
        }
        .
        .
        .
    }
"""

eval_device = '0'
eval_doc_path = './evaluation.txt'
eval_num_thread = 4

# 下面是一个构建 "eval_roots" 的例子:
eval_roots = dict()
datasets = ['MSRC', 'iCoSeg', 'CoSal2015', 'CoSOD3k', 'CoCA']

for dataset in datasets:
    roots = {'gt': '/mnt/jwd/data/{}/gt_bilinear_224/'.format(dataset), 
             'pred': './pred/{}/'.format(dataset)}
    eval_roots[dataset] = roots
# ------------ 示例结束 ------------

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = eval_device
    evaluate(roots=eval_roots, 
             doc_path=eval_doc_path,
             num_thread=eval_num_thread,
             pin=False)