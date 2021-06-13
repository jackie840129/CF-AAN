# encoding: utf-8
import logging

import torch
import torch.nn as nn
from ignite.engine import Engine

from utils.reid_metric import R1_mAP, R1_mAP_reranking
from ignite.contrib.handlers.tqdm_logger import ProgressBar


def create_supervised_evaluator(model, metrics,
                                device=None):
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def create_supervised_evaluator_with_mask(model, metrics,
                                device=None):
    if device:
        # if torch.cuda.device_count() > 1:
            # model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids ,masks = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data,masks)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def create_supervised_evaluator_with_mask_new_eval(model, metrics,
                                device=None):
   
    if device:
        # if torch.cuda.device_count() > 1:
            # model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, ambi, camids ,masks = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data,masks)
            return feat, pids, ambi, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def create_supervised_all_evaluator(model, metrics,seq_len,
                                device=None):
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        feats = []
        with torch.no_grad():
            data, pids, camids = batch
            iteration = data.shape[1]//seq_len
            for i in range(iteration):
                x = data[:,i*seq_len:(i+1)*seq_len,...]
                x = x.to(device) if torch.cuda.device_count() >= 1 else x
                feat = model(x)
                feats.append(feat)
            feats = torch.mean(torch.cat(feats,dim=0),dim=0,keepdim=True)
            return feats, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_supervised_all_evaluator_with_mask(model, metrics,seq_len,
                                device=None):
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        feats = []
        with torch.no_grad():
            data, pids, camids, masks = batch
            iteration = data.shape[1]//seq_len
            for i in range(iteration):
                x = data[:,i*seq_len:(i+1)*seq_len,...]
                mask = masks[:,i*seq_len:(i+1)*seq_len,...]
                x = x.to(device) if torch.cuda.device_count() >= 1 else x
                feat = model(x,mask)
                feats.append(feat)
            feats = torch.mean(torch.cat(feats,dim=0),dim=0,keepdim=True)
            return feats, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def create_supervised_all_evaluator_with_mask_new_eval(model, metrics,seq_len,
                                device=None):
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        feats = []
        with torch.no_grad():
            data, pids, ambi, camids, masks = batch
            iteration = data.shape[1]//seq_len
            for i in range(iteration):
                x = data[:,i*seq_len:(i+1)*seq_len,...]
                mask = masks[:,i*seq_len:(i+1)*seq_len,...]
                x = x.to(device) if torch.cuda.device_count() >= 1 else x
                feat = model(x,mask)
                feats.append(feat)
            feats = torch.mean(torch.cat(feats,dim=0),dim=0,keepdim=True)
            return feats, pids, ambi, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def inference(
        cfg,
        model,
        val_loader,
        num_query
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'no':
        print("Create evaluator")
        if 'test_all' in cfg.TEST.TEST_MODE:
            if len(val_loader.dataset.dataset[0]) == 4: # mask no new eval
                evaluator = create_supervised_all_evaluator_with_mask(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                        seq_len=cfg.INPUT.SEQ_LEN,device=device)
            elif len(val_loader.dataset.dataset[0]) == 6: # mask , new eval
                evaluator = create_supervised_all_evaluator_with_mask_new_eval(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,new_eval=True)},
                        seq_len=cfg.INPUT.SEQ_LEN,device=device)
            else:
                evaluator = create_supervised_all_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                        seq_len=cfg.INPUT.SEQ_LEN,device=device)
        else:
            if len(val_loader.dataset.dataset[0]) == 6: # mask , new eval
                evaluator = create_supervised_evaluator_with_mask_new_eval(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,new_eval=True)},
                        device=device)
            elif len(val_loader.dataset.dataset[0]) == 4 : # mask, no new eval
                evaluator = create_supervised_evaluator_with_mask(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                        device=device)
            else:
                evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    elif cfg.TEST.RE_RANKING == 'yes': # haven't implement with mask
        print("Create evaluator for reranking")
        if 'test_all' in cfg.TEST.TEST_MODE:
            evaluator = create_supervised_all_evaluator(model, metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                        seq_len=cfg.INPUT.SEQ_LEN,device=device)
        else:
            evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))

    pbar = ProgressBar(persist=True,ncols=120)
    pbar.attach(evaluator)

    evaluator.run(val_loader)
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
