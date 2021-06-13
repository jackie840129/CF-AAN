# encoding: utf-8
import torch


def train_collate_fn(batch):
    if len(batch[0]) == 4:
        imgs, pids, _,masks = zip(*batch)
        pids = torch.tensor(pids, dtype=torch.int64)
        return torch.stack(imgs, dim=0), pids ,torch.stack(masks,dim=0)
    imgs, pids, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    if len(batch[0]) == 4:
        imgs, pids, camids ,masks = zip(*batch)
        return torch.stack(imgs, dim=0), pids , camids, torch.stack(masks,dim=0)
    elif len(batch[0]) == 5 :
        imgs, pids, ambi, camids ,masks = zip(*batch)
        return torch.stack(imgs, dim=0), pids , ambi, camids, torch.stack(masks,dim=0)
    imgs, pids, camids = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids
