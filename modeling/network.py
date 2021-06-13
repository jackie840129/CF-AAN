# encoding: utf-8
import torch
from torch import nn
import torch.nn.functional as F

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.resnet_NL import ResNet_NL
from .backbones.ResNet import C2D_Axial_ResNet50


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class VNetwork(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice,temp,\
                    non_layers=[0,0,0,0], seq_len=6):
        super(VNetwork, self).__init__()
        self.seq_len = seq_len
        if model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50_NL':
            self.base = ResNet_NL(last_stride=last_stride,block=Bottleneck,
                                   layers=[3,4,6,3],non_layers=non_layers)
        elif model_name == 'resnet50_axial':
            self.base = C2D_Axial_ResNet50(seq_len=seq_len)

        if pretrain_choice == 'imagenet':
            if 'axial' not in model_name:
                self.base.load_param('',autoload='r50')
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.temp = temp
        self.model_name = model_name

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x,masks=None):
        b,t,c,h,w = x.shape

        if 'NL' in self.model_name:
            if self.temp == 'Done':
                x = self.base(x)
                _,c,h,w = x.shape
                if masks is not None:
                    global_feat = []
                    masks = masks.reshape(b*t,-1)
                    for i in range(x.shape[0]):
                        global_feat.append(self.gap(x[i,:,masks[i][0]:masks[i][1],masks[i][2]:masks[i][3]].unsqueeze(0)))
                    global_feat = torch.cat(global_feat,dim=0)
                    global_feat = torch.mean(global_feat.view(b,t,-1),dim=1)
                else:                    
                    global_feat = F.adaptive_avg_pool3d(x.view(b,t,c,h,w).permute(0,2,1,3,4),1)
                    global_feat = global_feat.view(b,-1)
            else:
                global_feat = self.gap(self.base(x))
                global_feat = global_feat.view(b*t,-1)  # flatten to (b*t, 2048)

        elif 'axial' in self.model_name:
            if masks is not None:
                global_feat = []
                masks = masks.reshape(b*t,-1)
                output = self.base(x.permute(0,2,1,3,4).contiguous()).permute(0,2,1,3,4).contiguous()
                b,t,c,h,w = output.shape
                output = output.view(b*t,c,h,w)
                for i in range(len(output)):
                    global_feat.append(self.gap(output[i,:,masks[i][0]:masks[i][1],masks[i][2]:masks[i][3]].unsqueeze(0)))
                global_feat = torch.cat(global_feat,dim=0)
                global_feat = torch.mean(global_feat.view(b,t,-1),dim=1)
            else:
                global_feat = self.base(x.permute(0,2,1,3,4).contiguous()).permute(0,2,1,3,4).contiguous()
                b,t,c,h,w = global_feat.shape
                global_feat = self.gap(global_feat.view(b*t,c,h,w))
                global_feat = torch.mean(global_feat.view(b,t,-1),dim=1)
        else:
            if masks is not None:
                global_feat = []
                masks = masks.reshape(b*t,-1)
                output = self.base(x.view(b*t,c,h,w))
                for i in range(len(output)):
                    global_feat.append(self.gap(output[i,:,masks[i][0]:masks[i][1],masks[i][2]:masks[i][3]].unsqueeze(0)))
                global_feat = torch.cat(global_feat,dim=0)
            else:
                global_feat = self.gap(self.base(x.view(b*t,c,h,w))) # (b*t, 2048, 1, 1)
            global_feat = global_feat.view(b*t,-1)  # flatten to (b*t, 2048)

        #### whether neck ####
        feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            if self.temp == 'avg':
                global_feat = torch.mean(global_feat.view(b,t,-1),dim=1)
                cls_score = torch.mean(cls_score.view(b,t,-1),dim=1) 
            
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.temp == 'avg':
                global_feat = torch.mean(global_feat.view(b,t,-1),dim=1)
                feat = torch.mean(feat.view(b,t,-1),dim=1)
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path,con=False):
        param_dict = torch.load(trained_path)['model']
        for i in param_dict:
            if 'classifier' in i and con == False:
                continue
            if 'bn_similarity' in i:
                if 'num' in i:
                    self.state_dict()[i].copy_(param_dict[i])
                else:
                    self.state_dict()[i][:param_dict[i].shape[0]].copy_(param_dict[i])
            elif 'bn_output' in i :
                if 'num' in i:
                    self.state_dict()[i].copy_(param_dict[i])
                else:
                    self.state_dict()[i][:param_dict[i].shape[0]].copy_(param_dict[i])
            else:
                self.state_dict()[i].copy_(param_dict[i])

