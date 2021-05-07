import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, help='datasets path')

args = parser.parse_args()
PATH = args.path
domain_list = ['Art', 'Clipart', 'Product', 'RealWorld']
for domain in domain_list:
    domain_path = os.path.join(PATH, domain)
    class_list = os.listdir(domain_path)
    for cls in tqdm(class_list):
        if cls == '.DS_Store':
                continue
        cls_path = os.path.join(domain_path, cls)
        img_list = os.listdir(cls_path)
        train_num = int(len(img_list)*0.7)

        train_path = os.path.join(domain_path, 'train', cls)
        test_path = os.path.join(domain_path, 'test', cls)
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        for i, img_name in enumerate(img_list):
            if img_name == '.DS_Store':
                continue
            img_path = os.path.join(cls_path, img_name)
            img = cv2.imread(img_path)
            if i<train_num:
                cv2.imwrite(os.path.join(train_path, img_name), img[:, :, ::-1])
            else:

                cv2.imwrite(os.path.join(test_path, img_name), img[:, :, ::-1])