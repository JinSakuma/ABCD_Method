import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
import glob
from PIL import Image 
from tqdm import tqdm
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset

from utils.utils import set_random_seeds, set_paths, get_cls_list, make_abcd_dataset, MyDataset, MyDataLoader
from utils.data import get_transform
from utils.trainer import train_classifier, train, val, test
# from models.models import DANN
from models.models_office31 import DANN, ResNetDANN
import wandb


parser = argparse.ArgumentParser()
# parser.add_argument('-m', '--method', type=str, default='softmax', help='softmax, or arcface')
# parser.add_argument('-o', '--output', type=str, default='logs/test/model3')
parser.add_argument('-o', '--output', type=str, default='logs/office31/v3/resnet50_lamda50_lr1e-4_sgd')
parser.add_argument('-e', '--epoch', type=int, default=100)
parser.add_argument('-g', '--gpuid', type=int, default=1)

args = parser.parse_args()

##########################################################################
# Config
##########################################################################
wandb_flg = True
if wandb_flg:
    exp_name = 'lambda10_lr1e-4_resnet50_sgd_pretrained'
    wandb.init(project="dann_office31_v3.0", entity='jsakuma', name=exp_name)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
set_random_seeds(0)

# task = 'mnist'
# root = '/mnt/aoni04/jsakuma/data'
# src_domain = 'mnist'
# tgt_domain = 'mnist-m'

task = 'office31'
root = '/mnt/aoni04/jsakuma/data/office31'
src_domain = 'amazon'
tgt_domain = 'webcam'


num_epochs = args.epoch
num_cls_train_epochs = 100
output_dir = args.output
batch_size=16
gamma = 10
hp_lambda = 10
    

# net = DANN(bn_flg=True)
# net = DANN()
net = ResNetDANN(model_name='resnet50', pretrained=True)

cls_criterion = nn.CrossEntropyLoss()
dom_criterion = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.005)
optimizer2 = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.005)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
cls_criterion.to(device)
dom_criterion.to(device)

if wandb_flg:
    wandb.watch(net)

##########################################################################
# Data
##########################################################################
src_train = set_paths(root, src_domain, 'train')
src_test = set_paths(root, src_domain, 'test')
tgt_train = set_paths(root, tgt_domain, 'train')
tgt_test = set_paths(root, tgt_domain, 'test')

cls_list = get_cls_list(root, 'amazon')
tgt_cls_num = int(len(cls_list)/2)
d_list = cls_list[tgt_cls_num:]

train_data = make_abcd_dataset(src_train, tgt_train, d_list, max_num=5000)
(X_a_train, y_a_train), (X_b_train, y_b_train), (X_c_train, y_c_train),(X_d_train, y_d_train) = train_data
test_data = make_abcd_dataset(src_test, tgt_test, d_list, max_num=800)
(X_a_test, y_a_test), (X_b_test, y_b_test), (X_c_test, y_c_test), (X_d_test, y_d_test) = test_data

X_abc_train = np.concatenate([X_a_train, X_b_train, X_c_train])
y_abc_train = np.concatenate([y_a_train, y_b_train, y_c_train])
d_abc_train = np.concatenate([np.zeros(len(y_a_train)), np.zeros(len(y_b_train)), np.ones(len(y_c_train))])
X_abc_test = np.concatenate([X_a_test, X_b_test, X_c_test])
y_abc_test = np.concatenate([y_a_test, y_b_test, y_c_test])
d_abc_test = np.concatenate([np.zeros(len(y_a_test)), np.zeros(len(y_b_test)), np.ones(len(y_c_test))])
d_d_train = np.ones(len(y_d_train))
d_d_test = np.ones(len(y_d_test))

transform_train, transform_test = get_transform(task)

#データセットの作成
ds_abc_train = MyDataset(X_abc_train, y_abc_train, d_abc_train, transform_train)
ds_d_train = MyDataset(X_d_train, y_d_train, d_d_train, transform_train)

ds_abc_test = MyDataset(X_abc_test, y_abc_test, d_abc_test, transform_test)
ds_d_test = MyDataset(X_d_test, y_d_test, d_d_test, transform_test)

#loaderの作成
loader_abc_train = MyDataLoader(ds_abc_train, batch_size=batch_size, shuffle=True)
loader_d_train = MyDataLoader(ds_d_train, batch_size=batch_size, shuffle=True)
loader_abc_test = MyDataLoader(ds_abc_test, batch_size=batch_size, shuffle=False)
loader_d_test = MyDataLoader(ds_d_test, batch_size=batch_size, shuffle=False)

##########################################################################
# train
##########################################################################
num_steps = len(loader_abc_train)
num_steps_val = len(loader_d_test)
os.makedirs(output_dir, exist_ok=True)

# train classifier only
print('##########################################')
print('train classifier only')
print('##########################################')
for epoch in range(num_cls_train_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_cls_train_epochs))
    print('-------------')

    epoch_loss  = train_classifier(num_steps, net, loader_abc_train, cls_criterion,
                                   dom_criterion, device, optimizer1)
    
#     val_loss, val_cls_loss, val_dom_loss, val_cls_acc, val_dom_acc = val(epoch, num_epochs, num_steps_val, net, loader_abc_test, cls_criterion, dom_criterion, device, hp_lambda, gamma)
    
# torch.save(net.state_dict(), os.path.join(output_dir, 'model_best_acc.pth'))


# train bath classifier and discriminator
print('##########################################')
print('train both classifier and discriminator')
print('##########################################')
os.makedirs(output_dir, exist_ok=True)
best_acc = 0.
for epoch in range(num_epochs+1):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-------------')
    
    if epoch == 0:
        train_loss, train_cls_loss, train_dom_loss, train_cls_acc, train_dom_acc = 0, 0, 0, 0, 0
        
        val_loss, val_cls_loss, val_dom_loss, val_cls_acc, val_dom_acc = val(epoch, num_epochs, num_steps_val, net, loader_abc_test, cls_criterion, dom_criterion, device, hp_lambda, gamma)
        
        test_d_loss, test_d_acc =test(net, loader_d_test, cls_criterion, device)

    else:
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                train_loss, train_cls_loss, train_dom_loss, train_cls_acc, train_dom_acc = train(epoch, num_epochs, num_steps, net, loader_abc_train, cls_criterion, dom_criterion, device, optimizer2, hp_lambda, gamma)
            elif phase == 'val':
                val_loss, val_cls_loss, val_dom_loss, val_cls_acc, val_dom_acc = val(epoch, num_epochs, num_steps_val, net, loader_abc_test, cls_criterion, dom_criterion, device, hp_lambda, gamma)
            else:
                test_d_loss, test_d_acc =test(net, loader_d_test, cls_criterion, device)
                print('test D loss: {:.3f}, acc: {:.3f}'.format(test_d_loss, test_d_acc))
                if test_d_acc>best_acc:
                        torch.save(net.state_dict(), os.path.join(output_dir, 'model_best_acc.pth'))
                        best_acc = test_d_acc
   
    if wandb_flg:
        wandb.log({
        "Train Loss": train_loss,
        "Train Cls Acc": train_cls_acc,
        "Train Dom Acc": train_cls_acc,
        "Train Cls Loss": train_cls_loss,
        "Train Dom Loss": train_dom_loss,
        "Valid Loss": val_loss,
        "Valid Cls Acc": val_cls_acc,
        "Valid Dom Acc": val_dom_acc,
        "Valid Cls Loss": val_cls_loss,
        "Valid Dom Loss": val_dom_loss,
        "Test D Loss":  test_d_loss,
        "Test D Acc":  test_d_acc,
        })

    
##########################################################################
# test
##########################################################################
net = DANN()
net.to(device)

model_path = os.path.join(output_dir, 'model_best_acc.pth')
net.load_state_dict(torch.load(model_path))

test_abc_loss, test_abc_acc =test(net, loader_abc_test, cls_criterion, device)
print('test ABC loss: {:.3f}, acc{:.3f}'.format(test_abc_loss, test_abc_acc))

test_d_loss, test_d_acc =test(net, loader_d_test, cls_criterion, device)
print('test D loss: {:.3f}, acc{:.3f}'.format(test_d_loss, test_d_acc))
