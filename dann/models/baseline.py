import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .grl import GradReverseLayer


        
        
class ResNet(nn.Module):
    def __init__(self, model_name, pretrained):
        super(ResNet, self).__init__()        

        if model_name == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            dim=512
        elif model_name == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            dim=2048
        
        self.classifier = base_model
        self.classifier.fc = nn.Linear(dim, 31)
        
     
    def forward(self, x):
        output = self.classifier(x)
        return output
    
    
class LeNet5(nn.Module):
    def __init__(self, bn_flg=False):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(48)
        self.mp = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout2d(0.5)
        
        self.fc1 = nn.Linear(48*5*5, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)
        self.bn3 = nn.BatchNorm1d(100)
        self.bn4 = nn.BatchNorm1d(100)
        
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
        
        x = x.view(-1, 48 * 5 * 5)
        
        x = self.fc1(x)
        if self.bn_flg:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if self.bn_flg:
            x = self.bn4(x)
        x = F.relu(x)
        output = self.fc3(x)
        
        return output
