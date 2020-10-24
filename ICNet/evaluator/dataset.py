from os import listdir
from os.path import join
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils import data
import torchvision.transforms as transforms

def build_file_paths(roots, suffixes):
    pred_base = roots['pred']
    gt_base = roots['gt']
    pred_suffix = suffixes['pred']
    gt_suffix = suffixes['gt']
    
    pred_paths = []
    gt_paths = []
    group_names = listdir(pred_base)
    for group_name in group_names:
        group_pred_names = list(filter(lambda name: name.endswith(pred_suffix), listdir(join(pred_base, group_name))))
        pred_paths += list(map(lambda pred_name: join(pred_base, group_name, pred_name), group_pred_names))
        gt_paths += list(map(lambda pred_name: join(gt_base, group_name, pred_name[:-len(pred_suffix)] + gt_suffix), group_pred_names))
    return gt_paths, pred_paths


class ImageData(data.Dataset):
    def __init__(self, roots, suffixes):
        gt_paths, pred_paths = build_file_paths(roots, suffixes)

        self.gt_paths = gt_paths
        self.pred_paths = pred_paths

    def __getitem__(self, item):
        gt = Image.open(self.gt_paths[item]).convert('L')
        pred = Image.open(self.pred_paths[item]).convert('L')

        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ])
        gt, pred = transform(gt), transform(pred)

        data_item = {}
        data_item['pred'] = pred
        data_item['gt'] = gt
        return data_item

    def __len__(self):
        return len(self.pred_paths)


def get_loader(roots, suffixes, batch_size, num_thread, pin=True):
    dataset = ImageData(roots, suffixes)
    data_loader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size, num_workers=num_thread, pin_memory=pin)
    return data_loader