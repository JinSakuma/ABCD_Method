import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob


def set_random_seeds(seed):
    """Set random seeds to ensure reproducibility."""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def set_paths(root, dataset_name, phase):
    cls_list = os.listdir(os.path.join(root, dataset_name, phase))
    cls_list = sorted(cls_list)
    keys = ['{}'.format(cls) for cls in cls_list]
    values = [[] for _ in range(len(cls_list))]
    path_dict = dict(zip(keys, values))
    for cls in cls_list:
        imgs_path = sorted(glob.glob(os.path.join(root, dataset_name, phase, '{}'.format(cls), '*.png')))
        if len(imgs_path)==0:
            imgs_path = sorted(glob.glob(os.path.join(root, dataset_name, phase, '{}'.format(cls), '*.jpg')))
        path_dict['{}'.format(cls)] += imgs_path
        
    return path_dict


def get_cls_list(root, dataset_name):
    cls_list = sorted(os.listdir(os.path.join(root, dataset_name, 'train')))
    return cls_list


def make_abcd_dataset(source_dict, target_dict, d_list, max_num=5000):
    X_a, y_a, X_b, y_b, X_c, y_c, X_d, y_d = [], [], [], [], [], [], [], []
    src_list = list(source_dict.values())
    src_lbl_list = list(source_dict.keys())
    for i, (src, lbl) in enumerate(zip(src_list, src_lbl_list)):
        if not lbl in d_list: 
            src = src[:max_num]
            X_a.extend(src)
            y_a.extend([i for _ in range(len(src))])
        else:
            src = src[:max_num]
            X_b.extend(src)
            y_b.extend([i for _ in range(len(src))])
            
        
    tgt_list = list(target_dict.values())
    tgt_lbl_list = list(target_dict.keys())
    for i, (tgt, lbl) in enumerate(zip(tgt_list, tgt_lbl_list)):
        if not lbl in d_list: 
            tgt = tgt[:max_num]
            X_c.extend(tgt)
            y_c.extend([i for _ in range(len(tgt))])
        else:
            tgt = tgt[:max_num]
            X_d.extend(tgt)
            y_d.extend([i for _ in range(len(tgt))])
            
    return (np.asarray(X_a), np.asarray(y_a)), (np.asarray(X_b), np.asarray(y_b)), (np.asarray(X_c), np.asarray(y_c)), (np.asarray(X_d), np.asarray(y_d))


class MyDataset(Dataset):
    def __init__(self, path, label, domain, transform):
        assert len(path) == len(label)
        self.image_path = path
        self.label = torch.LongTensor(label)
        self.domain = torch.LongTensor(domain)
        self.transform = transform
        
    def __getitem__(self, index):
        path = self.image_path[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), self.label[index], self.domain[index]

    def __len__(self):
        return len(self.image_path)
    

class MyDataLoader(Dataset):
    def __init__(self, dataset, shuffle=True, batch_size=1):
        super().__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.order = []
        if shuffle:
            self.order = np.random.permutation(len(self.dataset))
        else:
            self.order = np.arange(len(self.dataset))
        self.curr_idx = -1

    def __len__(self):
        return int(len(self.dataset)/self.batch_size)

    def __getitem__(self, idx):

        jdx = self.order[idx*self.batch_size:(idx+1)*self.batch_size]

        X_list, y_list, d_list = [], [], []
        for i in range(self.batch_size):
            X, y, d = self.dataset[jdx[i]]
            
            X_list.append(X)
            y_list.append(y)
            d_list.append(d)

        return torch.stack(X_list), torch.stack(y_list), torch.stack(d_list)

    def next(self):
        self.curr_idx += 1
        if self.curr_idx >= self.__len__():
            self.curr_idx = 0
        return self.__getitem__(self.curr_idx)

    def on_epoch_end(self):
        self.order = np.random.permutation(len(self.dataset))