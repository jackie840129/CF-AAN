# encoding: utf-8
import torch.nn.functional as F

from .triplet_loss import  TripletLoss,CrossEntropyLabelSmooth
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    # Creating Triplet
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        if cfg.SOLVER.SOFT_MARGIN:   margin = None
        else:  margin = cfg.SOLVER.MARGIN
        triplet = TripletLoss(margin)  # triplet loss
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    # Whether Label Smoothing
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)
    
    # Return loss_func
    loss_dict = {'triplet':0,'id_loss':0,'center':0}  # for logging
    if sampler == 'softmax':
        def loss_func(score, feat, target):
            id_loss = F.cross_entropy(score,target)
            loss_dict['id_loss'] = id_loss.item()
            return id_loss,loss_dict
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            metric = triplet(feat,target)[0]
            loss_dict['triplet'] = metric.item()
            return metric,loss_dict
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                metric = triplet(feat,target)[0]
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    id_loss = xent(score,target)
                else:
                    id_loss = F.cross_entropy(score,target)
                loss_dict['triplet'] = metric.item()
                loss_dict['id_loss'] = id_loss.item()
                return metric+id_loss,loss_dict
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


def make_loss_with_center(cfg, num_classes):    # modified by gu
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    else:
        feat_dim = 2048

    if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
        if cfg.SOLVER.SOFT_MARGIN:   margin = None
        else:  margin = cfg.SOLVER.MARGIN
        triplet = TripletLoss(margin)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    else:
        print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target):
        loss_dict = {'triplet':0,'id_loss':0,'center':0}
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            center = center_criterion(feat,target)
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                id_loss = xent(score, target)
            else:
                id_loss = F.cross_entropy(score, target)
            loss = cfg.SOLVER.CENTER_LOSS_WEIGHT * center + id_loss
            loss_dict['id_loss'] = id_loss.item()
            loss_dict['center'] = center.item()
            return loss,loss_dict

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            metric = triplet(feat,target)[0]
            center = center_criterion(feat,target)
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                id_loss = xent(score, target)
            else:
                id_loss = F.cross_entropy(score, target)
            loss = cfg.SOLVER.CENTER_LOSS_WEIGHT * center + id_loss + metric
            loss_dict['id_loss'] = id_loss.item()
            loss_dict['center'] = center.item()
            loss_dict['triplet'] = metric.item()
            return loss,loss_dict

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    return loss_func, center_criterion
