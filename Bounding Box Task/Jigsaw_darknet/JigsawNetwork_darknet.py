# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:57:01 2017

@author: Biagio Brattoli
"""
import torch
import torch.nn as nn
from torch import cat
import torch.nn.init as init

import sys
sys.path.append('Utils')
from Layers import LRN
import torch
import torch.nn as nn
from collections import OrderedDict
from darknet import darknet53

class Network(nn.Module):

    def __init__(self, classes=1000):
        super(Network, self).__init__()
        
        self.backbone = darknet53('')
        
        self.conv1_extra = nn.Sequential()
        self.conv1_extra.add_module('conv1', nn.Conv2d(1024, 256, 1, 1))
        self.conv1_extra.add_module('bn1',nn.BatchNorm2d(256))
        self.conv1_extra.add_module('leakyrelu1',nn.LeakyReLU(0.1))
        
        self.fc1_extra = nn.Sequential()
        self.fc1_extra.add_module('fc1',nn.Linear(256*3*3, 1024))
        self.fc1_extra.add_module('relu1',nn.ReLU(inplace=True))
        self.fc1_extra.add_module('drop1',nn.Dropout(p=0.5))

        self.fc2_extra = nn.Sequential()
        self.fc2_extra.add_module('fc2',nn.Linear(9*1024,4096))
        self.fc2_extra.add_module('relu2',nn.ReLU(inplace=True))
        self.fc2_extra.add_module('drop2',nn.Dropout(p=0.5))

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc3',nn.Linear(4096, classes))
        
        #self.apply(weights_init)

    def load(self,checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict and 'fc8' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])

    def save(self,checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def forward(self, x):
        B,T,C,H,W = x.size()
        x = x.transpose(0,1)

        x_list = []
        for i in range(9):
            z = self.backbone(x[i])[-1]
            z = self.conv1_extra(z)
            z = self.fc1_extra(z.view(B,-1))
            z = z.view([B,1,-1])
            x_list.append(z)

        x = cat(x_list,1)
        x = self.fc2_extra(x.view(B,-1))
        x = self.classifier(x)
        return x


def weights_init(model):
    if type(model) in [nn.Conv2d,nn.Linear]:
        nn.init.xavier_normal(model.weight.data)
        nn.init.constant(model.bias.data, 0.1)
    