import math
from torch.nn import functional as F
import numpy as np
import os
import torch
from torch import nn

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None,sub_sample=False, bn_layer=True,instance='soft',groups=1):
        super(NonLocalBlock, self).__init__()
        self.sub_sample = sub_sample
        self.instance = instance
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        self.groups = groups
        self.group_plane = self.inter_channels//self.groups
        ##### temporal operation in video re-id  #####
        conv_nd = nn.Conv3d
        max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
        bn = nn.BatchNorm3d
        ##############################################

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
        
    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1) # shape : (b , THW, c')

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1) # shape : (b, THW , c')
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1) # shape : (b , c', THW)
        
        f = torch.matmul(theta_x, phi_x)
        
        if self.instance == 'soft':
            f_div_C = F.softmax(f, dim=-1)
        elif self.instance == 'dot':
            f_div_C = f / f.shape[1]

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous() # shape : (b, c', THW)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:]) # shape : (b, c', T,  H, W)
        
        W_y = self.W(y) # shape : (b, c, t, h, w)
        z = W_y + x
        
        return z
