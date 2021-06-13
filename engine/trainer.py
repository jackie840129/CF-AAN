# encoding: utf-8
import logging

import torch
import torch.nn as nn
from torch.nn import DataParallel
# from engine.data_parallel import DataParallel  
# #self create dataparallel for unbalance GPU memory size
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer,global_step_from_engine
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from utils.reid_metric import R1_mAP


def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None):
    if device:
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
        model.to(device)
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img)
        loss,loss_dict = loss_fn(score, feat, target)
        loss.backward()
        optimizer.step()
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        loss_dict['loss'] = loss.item()
        return acc.item(),loss_dict

    return Engine(_update)

def create_supervised_trainer_with_mask(model, optimizer, loss_fn,
                              device=None):
    if device:
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
        model.to(device)
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target ,masks = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img,masks)
        loss,loss_dict = loss_fn(score, feat, target)
        loss.backward()
        optimizer.step()
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        loss_dict['loss'] = loss.item()
        return acc.item(),loss_dict

    return Engine(_update)

def create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cetner_loss_weight,
                              device=None):
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img)
        loss,loss_dict = loss_fn(score, feat, target)
        # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))
        loss.backward()
        optimizer.step()
        for param in center_criterion.parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        optimizer_center.step()

        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        loss_dict['loss'] = loss.item()
        return acc.item(),loss_dict

    return Engine(_update)

# +
def create_supervised_evaluator(model, metrics,
                                device=None):
    if device:
        # if torch.cuda.device_count() > 1:
            # model = nn.DataParallel(model)
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

# -

def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    # checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    # Create 1. trainer  2. evaluator 3. checkpointer 4. timer 5. pbar
    if len(train_loader.dataset.dataset[0]) == 4 : #train with mask
        trainer = create_supervised_trainer_with_mask(model, optimizer, loss_fn, device=device)
        if cfg.TEST.NEW_EVAL == False:
            evaluator = create_supervised_evaluator_with_mask(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
        else:
            evaluator = create_supervised_evaluator_with_mask_new_eval(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,new_eval=True)}, device=device)
    else: # no mask
        trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
        if cfg.TEST.NEW_EVAL == False:
            evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
        else:
            raise NotImplementedError
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, n_saved=1, require_empty=False,\
                                    score_function=lambda x : x.state.metrics['r1_mAP'][1],\
                                    global_step_transform=global_step_from_engine(trainer))
    timer = Timer(average=True)
    tpbar = ProgressBar(persist=True,ncols=120)
    epbar = ProgressBar(persist=True,ncols=120)
    #############################################################
    evaluator.add_event_handler(Events.EPOCH_COMPLETED(every=1), checkpointer, \
        {'model': model,'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
    tpbar.attach(trainer)
    epbar.attach(evaluator)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_acc')
    RunningAverage(output_transform=lambda x: x[1]['loss']).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]['triplet']).attach(trainer, 'avg_trip')


    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_learning_rate(engine):
        # if engine.state.epoch == 1:
            # scheduler.step()
        scheduler.step()


    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Total Loss : {:.3f}, Triplet Loss : {:.3f}, Acc : {:.3f}, Base Lr : {:.2e}'
                    .format(engine.state.epoch, engine.state.metrics['avg_loss'],engine.state.metrics['avg_trip'],
                            engine.state.metrics['avg_acc'],scheduler.get_last_lr()[0]))
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            # evaluator.state.epoch = trainer.state.epoch
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)


def do_train_with_center(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    # checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, None, n_saved=10, require_empty=False)
    timer = Timer(average=True)
    pbar = ProgressBar(persist=True,ncols=120)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=checkpoint_period), checkpointer, {'model': model,
                                                                     'optimizer': optimizer,
                                                                     'center_param': center_criterion,
                                                                     'optimizer_center': optimizer_center})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
    pbar.attach(trainer)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_acc')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]['triplet']).attach(trainer, 'avg_trip')
    RunningAverage(output_transform=lambda x: x[1]['center']).attach(trainer, 'avg_center')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_learning_rate(engine):
        scheduler.step()
    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Total Loss : {:.3f}, Triplet Loss : {:.3f}, Center Loss , Acc : {:.3f}, Base Lr : {:.2e}'
                    .format(engine.state.epoch, engine.state.metrics['avg_loss'],engine.state.metrics['avg_trip'],
                            engine.state.metrics['avg_center'],engine.state.metrics['avg_acc'],scheduler.get_lr()[0]))
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)
