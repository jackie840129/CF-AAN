# encoding: utf-8
import numpy as np
from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset,VideoDataset
from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid
from .transforms import build_transforms_ST


def make_data_loader(cfg):
    ##### build transform #####
    train_spatial_transforms , _ = build_transforms_ST(cfg, is_train=True)
    val_spatial_transforms, val_temporal_transforms = build_transforms_ST(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    ##### init dataset-specific object #####
    if cfg.MODEL.SETTING == 'video':
        dataset = init_dataset(cfg.DATASETS.NAMES[0], root=cfg.DATASETS.ROOT_DIR,min_seq_len=cfg.INPUT.MIN_SEQ_LEN,new_eval=cfg.TEST.NEW_EVAL)
    else:
        raise NotImplementedError()

    num_classes = dataset.num_train_pids
    ##### create real pytorch Dataset #####
    if cfg.MODEL.SETTING == 'video':
        train_set = VideoDataset(dataset.train,cfg.INPUT.SEQ_LEN, cfg.INPUT.SAMPLE, train_spatial_transforms, None, mode='train')
    else:
        raise NotImplementedError()
    
    ##### create dataloader #####
    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            worker_init_fn= lambda _:np.random.seed(),
            sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn,drop_last=True
        )
    if cfg.MODEL.SETTING == 'video':
        val_set = VideoDataset(dataset.query + dataset.gallery, cfg.INPUT.SEQ_LEN,cfg.INPUT.SAMPLE, val_spatial_transforms,val_temporal_transforms, mode=cfg.TEST.TEST_MODE)
    else:
        raise NotImplementedError()

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes
