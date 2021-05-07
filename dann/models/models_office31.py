import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
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
#         print(x.shape)
        output = x.view(-1, 48 * 53 * 53)
        
        return output
    
    
class Classifier(nn.Module):
    def __init__(self, bn_flg=False):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(48*53*53, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 31)
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
        self.fc1 = nn.Linear(48*53*53, 100)
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
        
        
class ResNetExtractor(nn.Module):
    def __init__(self, base_model, dim=512):
        super(ResNetExtractor, self).__init__()
        self.base_model = base_model
        self.extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.dim = dim
        
    def forward(self, x):
        output = self.extractor(x)
        output = output.view(-1, self.dim * 1 * 1)
        return output 
    
    
class ResNetClassifier(nn.Module):
    def __init__(self, dim=512):
        super(ResNetClassifier, self).__init__()
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 31)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        output = self.fc3(x)
        
        return output

    
class ResNetDiscriminator(nn.Module):
    def __init__(self, dim=512):
        super(ResNetDiscriminator, self).__init__()
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(dim, 100)
        self.fc2 = nn.Linear(100, 2)
        self.bn1 = nn.BatchNorm1d(100)

    def forward(self, x, hp_lambda):
        x = GradReverseLayer.grad_reverse(x, hp_lambda)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        output = self.fc2(x)
        
        return output

        
        
class ResNetDANN(nn.Module):
    def __init__(self, model_name, pretrained):
        super(ResNetDANN, self).__init__()        
#         self.bn_flg = bn_flg

        if model_name == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            dim=512
        elif model_name == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            dim=2048
            
        self.extractor = ResNetExtractor(base_model, dim=dim)
        self.classifier = ResNetClassifier(dim=dim)
        self.discriminator = ResNetDiscriminator(dim=dim)
