import torch
import torch.nn as nn
import numpy as np

class fc_layer(nn.Module):
    def __init__(self, in_dim, out_dim, linear=False, dropout=False):
        super(fc_layer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        if linear:
            lecunn_uniform(self.linear)
        else:
            glorot_uniform(self.linear)
        self.dropout = nn.Dropout() if dropout else nn.Sequential()
        self.act = nn.ReLU(inplace=True) if not linear else nn.Sequential()

    def forward(self, x):
        out = self.linear(x)
        out = self.act(out) 
        return self.dropout(out)

class conv_layer(nn.Module):
    def __init__(self, in_dim, out_dim, k=3, stride=1):
        super(conv_layer, self).__init__()
        padding = (k - 1) // 2
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=k, padding=padding, stride=stride, bias=True)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        conv = self.conv(x)
        return self.act(conv)

class residual_layer(nn.Module):
    def __init__(self, in_dim, out_dim, k=3, with_bn=True):
        super(residual_layer, self).__init__()
        padding = (k - 1) // 2 
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=k, padding=padding, bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=True)
            )
        
    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return self.shortcut(x) + relu
        
def glorot_uniform(layer):
    fan_in, fan_out = layer.in_features, layer.out_features
    limit = np.sqrt(6. / (fan_in + fan_out))
    layer.weight.data.uniform_(-limit, limit)

def lecunn_uniform(layer):
    fan_in, fan_out = layer.in_features, layer.out_features
    limit = np.sqrt(3. / fan_in)
    layer.weight.data.uniform_(-limit, limit)
