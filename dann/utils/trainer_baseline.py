import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def train(num_steps, net, dataloader, cls_criterion, device, optimizer):
    
    dataloader.on_epoch_end()
    net.train()

    epoch_loss = 0.0
    correct = 0.0
    train_cnt = 0
    for batch_idx, (X, y, d) in zip(tqdm(range(num_steps)), dataloader):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # classification
        cls_pred = net(X)
        cls_loss = cls_criterion(cls_pred, y)
        
        cls_loss.backward()
        optimizer.step()
        cls_loss = cls_loss.item()

        epoch_loss += cls_loss
        loss = 0
        train_cnt += X.shape[0]
        
        cls_pred = cls_pred.argmax(dim=1, keepdim=True)
        correct += cls_pred.eq(y.view_as(cls_pred)).sum().item()

    epoch_loss = epoch_loss / train_cnt
    cls_acc = correct / float(train_cnt)
    print('loss: {:.3f}, cls acc: {:.3f}'.format(epoch_loss, cls_acc))

    return epoch_loss, cls_acc


def val(num_steps, net, dataloader, cls_criterion,device):
   
    dataloader.on_epoch_end()
    net.eval()

    epoch_loss = 0.0
    correct = 0.0
    train_cnt = 0
    with torch.no_grad():
        for batch_idx, (X, y, d) in zip(tqdm(range(num_steps)), dataloader):
            X, y = X.to(device), y.to(device)

            # classification
            cls_pred = net(X)
            cls_loss = cls_criterion(cls_pred, y)
            cls_loss = cls_loss.item()

            epoch_loss += cls_loss
            loss = 0
            train_cnt += X.shape[0]

            cls_pred = cls_pred.argmax(dim=1, keepdim=True)
            correct += cls_pred.eq(y.view_as(cls_pred)).sum().item()

    epoch_loss = epoch_loss / train_cnt
    cls_acc = correct / float(train_cnt)
    print('loss: {:.3f}, cls acc: {:.3f}'.format(epoch_loss, cls_acc))

    return epoch_loss, cls_acc




def test(net, dataloader, criterion, device):

    net.eval()
    epoch_loss = 0.0
    correct = 0.0
    train_cnt = 0
    best_acc = 0

    with torch.no_grad():
        for X, y, d in tqdm(dataloader):
            X = X.to(device)
            y = y.to(device)

            # classification
            cls_pred = net(X)

            loss = criterion(cls_pred, y)
            loss = loss.item()

            epoch_loss += loss
            loss = 0
            train_cnt += X.shape[0]
            
            output = F.softmax(cls_pred, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()

        epoch_loss = epoch_loss / float(train_cnt)
        acc = correct / float(train_cnt)

    return epoch_loss, acc