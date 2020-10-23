import os
from evaluator.evaluator import evaluate_dataset
from utils import write_doc

"""
* Note:
    The evaluation codes in "./evaluator/" are implemented in PyTorch (GPU-version) for acceleration.

    Since some GTs (e.g. in "Cosal2015" dataset) are of too large original sizes to be evaluated on GPU with limited memory 
    (our "TITAN Xp" runs out of 12G memory when computing F-measure), the input prediction map and corresponding GT 
    are resized to 224*224 by our evaluation codes before computing metrics.
"""

"""
evaluate:
    Given predictions, compute multiple metrics (max F-measure, S-measure and MAE).
    The evaluation results are saved in "doc_path".
"""
def evaluate(roots, doc_path, num_thread, pin):
    datasets = roots.keys()
    for dataset in datasets:
        # Evaluate predictions of "dataset".
        results = evaluate_dataset(roots=roots[dataset], 
                                   dataset=dataset,
                                   batch_size=1, 
                                   num_thread=num_thread, 
                                   demical=True,
                                   suffixes={'gt': '.png', 'pred': '.png'},
                                   pin=pin)
        
        # Save evaluation results.
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
Evaluation settings (used for "evaluate.py"):

eval_device:
    Index of the GPU used for evaluation.

eval_doc_path:
    Path of the file (".txt") used to save the evaluation results.

eval_roots:
    A dictionary including multiple sub-dictionary, 
    each sub-dictionary contains the GT and prediction folder paths of a specific dataset.
    Format:
    eval_roots = {
        name of dataset_1: {
            'gt': GT folder path of dataset_1,
            'pred': prediction folder path of dataset_1
        },
        name of dataset_2: {
            'gt': GT folder path of dataset_2,
            'pred': prediction folder path of dataset_2
        }
        .
        .
        .
    }
"""

eval_device = '0'
eval_doc_path = './evaluation.txt'
eval_num_thread = 4

# An example to build "eval_roots".
eval_roots = dict()
datasets = ['MSRC', 'iCoSeg', 'CoSal2015', 'CoSOD3k', 'CoCA']

for dataset in datasets:
    roots = {'gt': '/mnt/jwd/data/{}/gt_bilinear_224/'.format(dataset), 
             'pred': './pred/{}/'.format(dataset)}
    eval_roots[dataset] = roots
# ------------- end -------------

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = eval_device
    evaluate(roots=eval_roots, 
             doc_path=eval_doc_path,
             num_thread=eval_num_thread,
             pin=False)