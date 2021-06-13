import torch
import math
import copy
import torchvision
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch.nn import functional as F

from .SA import inflate
from .SA import AP3D
from .SA import NonLocal
from .SA import SelfAttn


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)      


class Bottleneck3D(nn.Module):
    def __init__(self, bottleneck2d, block, inflate_time=False, temperature=4, contrastive_att=True):
        super(Bottleneck3D, self).__init__()

        self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)
        if inflate_time == True:
            self.conv2 = block(bottleneck2d.conv2, temperature=temperature, contrastive_att=contrastive_att)
        else:
            self.conv2 = inflate.inflate_conv(bottleneck2d.conv2, time_dim=1)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)
        self.conv3 = inflate.inflate_conv(bottleneck2d.conv3, time_dim=1)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)
        self.relu = nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = self._inflate_downsample(bottleneck2d.downsample)
        else:
            self.downsample = None

    def _inflate_downsample(self, downsample2d, time_stride=1):
        downsample3d = nn.Sequential(
            inflate.inflate_conv(downsample2d[0], time_dim=1, 
                                 time_stride=time_stride),
            inflate.inflate_batch_norm(downsample2d[1]))
        return downsample3d

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet503D(nn.Module):
    def __init__(self, block, c3d_idx, nl_idx, sa_idx, temperature=4, contrastive_att=True, seq_len=6,**kwargs):
        super(ResNet503D, self).__init__()

        self.block = block
        self.temperature = temperature
        self.contrastive_att = contrastive_att
        self.inplanes = 64 
        self.seq_len = seq_len

        resnet2d = torchvision.models.resnet50(pretrained=True)
        resnet2d.layer4[0].conv2.stride=(1, 1)
        resnet2d.layer4[0].downsample[0].stride=(1, 1) 
        
        ############ STEM ###################
        self.conv1 = inflate.inflate_conv(resnet2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(resnet2d.maxpool, time_dim=1)
        #####################################

        self.layer1 = self._inflate_reslayer(resnet2d.layer1, c3d_idx=c3d_idx[0], \
                                             nl_idx=nl_idx[0], sa_idx= sa_idx[0],in_channels=256,ks=[64,32,seq_len])
        self.layer2 = self._inflate_reslayer(resnet2d.layer2, c3d_idx=c3d_idx[1], \
                                             nl_idx=nl_idx[1], sa_idx=sa_idx[1],in_channels=512,ks=[32,16,seq_len])
        self.layer3 = self._inflate_reslayer(resnet2d.layer3, c3d_idx=c3d_idx[2], \
                                             nl_idx=nl_idx[2], sa_idx=sa_idx[2],in_channels=1024,ks=[16,8,seq_len])
        self.layer4 = self._inflate_reslayer(resnet2d.layer4, c3d_idx=c3d_idx[3], \
                                             nl_idx=nl_idx[3], sa_idx=sa_idx[3],in_channels=2048,ks=[16,8,seq_len])

    def _inflate_reslayer(self, reslayer2d, c3d_idx, nl_idx=[], sa_idx=[],in_channels=0,ks=[64,32,1]):
        reslayers3d = []
        for i,layer2d in enumerate(reslayer2d):
            if i not in c3d_idx: # normal 2D convolution
                layer3d = Bottleneck3D(layer2d, AP3D.C2D, inflate_time=False)
            else: # (AP)I3D, (AP)P3D-A,B,C
                layer3d = Bottleneck3D(layer2d, self.block, inflate_time=True, \
                                       temperature=self.temperature, contrastive_att=self.contrastive_att)
            reslayers3d.append(layer3d)

            if (i in nl_idx) and (i not in sa_idx):
                non_local_block = NonLocal.NonLocalBlock3D(in_channels, sub_sample=True)
                reslayers3d.append(non_local_block)
            elif (i in sa_idx) and (i not in nl_idx):
                if ks[0] == 32:
                    sa_block = SelfAttn.AxialBlock(in_channels,inter_channel=None,kernel_size=ks,granularity=4,groups=8,positional='r_qkv',order='hwt')
                else:
                    sa_block = SelfAttn.AxialBlock(in_channels,inter_channel=None,kernel_size=ks,granularity=4,groups=8,positional='r_qkv',order='hwt')
                reslayers3d.append(sa_block)
            elif (i in sa_idx) and (i in nl_idx):
                raise ValueError("can not use nl and sa at the same time!")
        return nn.Sequential(*reslayers3d)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def AP3DResNet50(num_classes, **kwargs):
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[],[],[]]

    return ResNet503D(num_classes, AP3D.APP3DC, c3d_idx, nl_idx, **kwargs)

def P3D_ResNet50(**kwargs):
    c3d_idx = [[],[0,2],[0,2,4],[]]
    nl_idx = [[],[],[],[]]
    sa_idx = [[],[],[],[]]
    return ResNet503D(AP3D.P3DC, c3d_idx, nl_idx, sa_idx, **kwargs)

def P3D_Axial_ResNet50(**kwargs):
    c3d_idx = [[],[0,1],[0,1,2],[]]
    nl_idx = [[],[],[],[]]
    sa_idx = [[],[2,3],[3,4,5],[]]
    return ResNet503D(AP3D.P3DC, c3d_idx, nl_idx, sa_idx, **kwargs)

def C2D_Axial_ResNet50(**kwargs):
    c3d_idx = [[],[],[],[]]
    nl_idx = [[],[],[],[]]
    sa_idx = [[],[2,3],[3,4,5],[]]
    return ResNet503D(AP3D.APP3DC, c3d_idx, nl_idx, sa_idx, **kwargs)


def C2DResNet50(num_classes, **kwargs):
    c3d_idx = [[],[],[],[]]
    nl_idx = [[],[],[],[]]
    return ResNet503D(num_classes, AP3D.APP3DC, c3d_idx, nl_idx, sa_idx, **kwargs)

def C2DNLResNet50(num_classes, **kwargs):
    c3d_idx = [[],[],[],[]]
    nl_idx = [[],[2, 3],[3, 4, 5],[]]

    return ResNet503D(num_classes, AP3D.APP3DC, c3d_idx, nl_idx, **kwargs)

def AP3DNLResNet50(num_classes, **kwargs):
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[1, 3],[1, 3, 5],[]]

    return ResNet503D(num_classes, AP3D.APP3DC, c3d_idx, nl_idx, **kwargs)
