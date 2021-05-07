import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_classifier(num_steps, net, dataloader, cls_criterion, dom_criterion,
          device, optimizer
          ):
    
    dataloader.on_epoch_end()
    net.train()
#     scheduler.step()
#     print('lr:{}'.format(scheduler.get_lr()[0]))

    epoch_loss = 0.0
    epoch_cls_loss = 0.0
    correct = 0.0
    train_cnt = 0
    for batch_idx, (X, y, d) in zip(tqdm(range(num_steps)), dataloader):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # feature extraction
        feat = net.extractor(X)
        
        # classification
        cls_pred = net.classifier(feat)
        
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

    return epoch_loss


def train(epoch, num_epochs, num_steps, net, dataloader, cls_criterion, dom_criterion,
          device, optimizer, hp_lambda, gamma
          ):
    
    dataloader.on_epoch_end()
    net.train()
#     scheduler.step()
#     print('lr:{}'.format(scheduler.get_lr()[0]))

    epoch_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_dom_loss = 0.0
    cls_correct = 0.
    dom_correct = 0.
    train_cnt = 0.
    for i, (X, y, d) in zip(tqdm(range(num_steps)), dataloader):
        X, y, d = X.to(device), y.to(device), d.to(device)
        
        # set params
#         p = float(i+epoch*num_steps) / num_epochs / num_steps
#         hp_lambda = 2. / (1. + np.exp(-gamma * p)) - 1
#         print('hp_lambda: {}'.format(hp_lambda))
#         lr = 0.01 / (1. + 10 * p) ** 0.75
#         print('lr: {}'.format(lr))
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
        
        optimizer.zero_grad()
        
        # feature extraction
        feat = net.extractor(X)
        
        # classification
        cls_pred = net.classifier(feat)
  
        cls_loss = cls_criterion(cls_pred, y)
        
        # domain discrimination
        dom_pred = net.discriminator(feat, hp_lambda)
        dom_loss = dom_criterion(dom_pred, d)
        
        # total loss backward
        loss = cls_loss + dom_loss
        loss.backward()
        optimizer.step()
        loss = loss.item()

        epoch_loss += loss
        epoch_cls_loss += cls_loss.item()
        epoch_dom_loss += dom_loss.item()
        loss = 0
        cls_loss = 0
        dom_loss = 0
        
        train_cnt += X.shape[0]
        cls_pred = cls_pred.argmax(dim=1, keepdim=True)
        cls_correct += cls_pred.eq(y.view_as(cls_pred)).sum().item()
        
        dom_pred = dom_pred.argmax(dim=1, keepdim=True)
        dom_correct += dom_pred.eq(d.view_as(dom_pred)).sum().item()

    epoch_cls_loss = epoch_cls_loss / train_cnt
    epoch_dom_loss = epoch_dom_loss / train_cnt
    epoch_loss = epoch_cls_loss + epoch_dom_loss
    cls_acc = cls_correct / float(train_cnt)
    dom_acc = dom_correct / float(train_cnt)
    print('train')
    print('loss: {:.3f}, cls loss: {:.3f}, dom loss: {:.3f}'.format(epoch_loss, epoch_cls_loss, epoch_dom_loss))
    print('cls_acc: {:.3f}, dom_acc: {:.3f}'.format(cls_acc, dom_acc))


    return epoch_loss, epoch_cls_loss, epoch_dom_loss, cls_acc, dom_acc


def val(epoch, num_epochs, num_steps, net, dataloader, cls_criterion, dom_criterion,
        device, hp_lambda, gamma
        ):

    net.eval()
    epoch_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_dom_loss = 0.0
    cls_correct = 0.
    dom_correct = 0.
    train_cnt = 0.
    with torch.no_grad():
        for i, (X, y, d) in zip(tqdm(range(num_steps)), dataloader):
            X, y, d = X.to(device), y.to(device), d.to(device)
            
            # set params
#             p = float(i+epoch*num_steps) / num_epochs / num_steps
#             hp_lambda = 2. / (1. + np.exp(-gamma * p)) - 1
#             print('hp_lambda: {}'.format(hp_lambda))
#             lr = 0.01 / (1. + 10 * p) ** 0.75
#             print('lr: {}'.format(lr))
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr
                
            #feature extraction
            feat = net.extractor(X)

            # classification
            cls_pred = net.classifier(feat)

            cls_loss = cls_criterion(cls_pred, y)

            # domain discrimination
            dom_pred = net.discriminator(feat, hp_lambda)
            dom_loss = dom_criterion(dom_pred, d)

            # total loss backward
            loss = cls_loss + dom_loss
            loss = loss.item()

            epoch_loss += loss
            epoch_cls_loss += cls_loss.item()
            epoch_dom_loss += dom_loss.item()
            loss = 0
            cls_loss = 0
            dom_loss = 0

            train_cnt += X.shape[0]
            cls_pred = cls_pred.argmax(dim=1, keepdim=True)
            cls_correct += cls_pred.eq(y.view_as(cls_pred)).sum().item()

            dom_pred = dom_pred.argmax(dim=1, keepdim=True)
            dom_correct += dom_pred.eq(d.view_as(dom_pred)).sum().item()

        epoch_cls_loss = epoch_cls_loss / train_cnt
        epoch_dom_loss = epoch_dom_loss / train_cnt
        epoch_loss = epoch_cls_loss + epoch_dom_loss
        cls_acc = cls_correct / float(train_cnt)
        dom_acc = dom_correct / float(train_cnt)
        print('val')
        print('loss: {:.3f}, cls loss: {:.3f}, dom loss: {:.3f}'.format(epoch_loss, epoch_cls_loss, epoch_dom_loss))
        print('cls_acc: {:.3f}, dom_acc: {:.3f}'.format(cls_acc, dom_acc))

    return epoch_loss, epoch_cls_loss, epoch_dom_loss, cls_acc, dom_acc


def test(net, dataloader, criterion, device
        ):

    net.eval()
    epoch_loss = 0.0
    correct = 0.0
    train_cnt = 0
    best_acc = 0

    with torch.no_grad():
        for X, y, d in tqdm(dataloader):
            X = X.to(device)
            y = y.to(device)
            
            # feature extraction
            feat = net.extractor(X)

            # classification
            cls_pred = net.classifier(feat)

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