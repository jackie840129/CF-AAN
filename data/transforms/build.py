# encoding: utf-8
import torchvision.transforms as T

from .transforms import RandomErasing
from .temporal_transforms import TemporalBeginCrop


def build_transforms_ST(cfg,is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform_list = [T.Resize(cfg.INPUT.SIZE_TRAIN)]
        if cfg.INPUT.IF_FLIP == True:
            transform_list.append(T.RandomHorizontalFlip(p=cfg.INPUT.PROB))
        if cfg.INPUT.IF_CROP == True:
            transform_list.append(T.Pad(cfg.INPUT.PADDING))
            transform_list.append(T.RandomCrop(cfg.INPUT.SIZE_TRAIN))
        transform_list += [T.ToTensor(),normalize_transform]
        if cfg.INPUT.IF_RE == True:
            transform_list.append(RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN))
        spatial_transform = T.Compose(transform_list)
        temporal_transforms = None
    else:
        spatial_transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])
        temporal_transforms = TemporalBeginCrop(size=cfg.INPUT.SEQ_LEN)
    return spatial_transform,temporal_transforms
