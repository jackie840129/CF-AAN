# encoding: utf-8
from .network import VNetwork


def build_model(cfg, num_classes):
    if cfg.MODEL.SETTING == 'video':
        model = VNetwork(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, \
                         cfg.MODEL.PRETRAIN_CHOICE, cfg.MODEL.TEMP,cfg.MODEL.NON_LAYERS,cfg.INPUT.SEQ_LEN)
        return model
    else:
        raise NotImplementedError()