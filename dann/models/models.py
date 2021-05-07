import torch
import torch.nn as nn
import torch.nn.functional as F
from .grl import GradReverseLayer


class Extractor(nn.Module):
    def __init__(self, bn_flg=False):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(48)
        self.mp = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout2d(0.5)
        
        self.bn_flg = bn_flg

    def forward(self, x):
        x = self.conv1(x)
        if self.bn_flg:
            x = self.bn1(x)
        x = self.mp(x)
        x = F.relu(x)
        x = self.conv2(x)
        if self.bn_flg:
            x = self.bn2(x)
        x = self.dropout(x)
        x = self.mp(x)
        x = F.relu(x)
        output = x.view(-1, 48 * 5 * 5)
        
        return output
    
    
class Classifier(nn.Module):
    def __init__(self, bn_flg=False):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(48*5*5, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)
        self.dropout = nn.Dropout2d(0.5)
        
        self.bn_flg = bn_flg

    def forward(self, x):
        x = self.fc1(x)
        if self.bn_flg:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if self.bn_flg:
            x = self.bn2(x)
        x = F.relu(x)
        output = self.fc3(x)
        
        return output

    
class Discriminator(nn.Module):
    def __init__(self, bn_flg=False):
        super(Discriminator, self).__init__()
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(48*5*5, 100)
        self.fc2 = nn.Linear(100, 2)
        self.bn1 = nn.BatchNorm1d(100)
        
        self.bn_flg = bn_flg

    def forward(self, x, hp_lambda):
        x = GradReverseLayer.grad_reverse(x, hp_lambda)
        x = self.fc1(x)
        if self.bn_flg:
            x = self.bn1(x)
        x = F.relu(x)
        output = self.fc2(x)
        
        return output


class DANN(nn.Module):
    def __init__(self, bn_flg=False):
        super(DANN, self).__init__()        
        self.bn_flg = bn_flg
        self.extractor = Extractor(self.bn_flg)
        self.classifier = Classifier(self.bn_flg)
        self.discriminator = Discriminator(self.bn_flg)

#     def forward(self, x, hp_lambda):
#         feat = self.extractor(x)
#         cls = self.classifier(feat)
#         domain = self.discriminator(feat, hp_lambda)
        
#         return feat, cls, domain
    
#     def train_classifier(self, x):
#         feat = self.extractor(x)
#         cls = self.classifier(feat)
        
#         return cls





