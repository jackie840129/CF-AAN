# encoding: utf-8
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import random


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path

class VideoDataset(Dataset):
    """Video Person ReID Dataset"""

    def __init__(self,dataset,seq_len=8,sample='RRS',spatial_transform=None, temporal_transform=None,mode='test'):
        self.dataset = dataset
        self.mask = len(dataset[0]) == 4 or len(dataset[0])==6
        self.new_eval = len(dataset[0]) == 5 or len(dataset[0])== 6
        self.seq_len = seq_len
        self.sample = sample
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.mode = mode
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        if self.mask and not self.new_eval:
            img_paths, pid, cam ,mask = self.dataset[idx]
        elif self.mask and self.new_eval:
            img_paths, _,  pid, ambi, cam ,mask = self.dataset[idx]
        elif not self.mask and self.new_eval:
            raise NotImplementedError
        else:
            img_paths, pid, cam = self.dataset[idx]
        
        num = len(img_paths)
        indices = np.arange(0,num).astype(np.int32)

        # Temporal Sample Methods #
        if self.sample == 'RRS' and self.mode != 'test_0':
            
            num_pads = 0 if num%self.seq_len==0 else self.seq_len - num%self.seq_len
            indices = np.concatenate([indices,np.ones(num_pads).astype(np.int32)*(num-1)])
            assert len(indices) %self.seq_len == 0

            indices_pool = np.split(indices,self.seq_len)
            sampled_indices = []

            if self.mode == 'train':
                for part in indices_pool:
                    sampled_indices.append(np.random.choice(part,1)[0])
            elif self.mode == 'test_all_sampled':
                sampled_indices = np.vstack(indices_pool).T.flatten()
            elif self.mode == 'test_all_continuous':
                sampled_indices = np.vstack(indices_pool).flatten()
            else : 
                for part in indices_pool:
                    sampled_indices.append(part[0])

        elif self.mode == 'test_0':
            sampled_indices = self.temporal_transform(indices)
        ################################

        imgs = []
        for index in sampled_indices:
            img_path = img_paths[index]
            img = read_image(img_path)
            if self.spatial_transform is not None:
                img = self.spatial_transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs,dim=0)

        if self.mode == 'train':
            flip_prob = random.random()
            if flip_prob > 0.5:
                imgs = torch.flip(imgs,dims=[3])

        if self.mask:
            sampled_mask = mask[sampled_indices,:]
            if self.mode == 'train' and flip_prob > 0.5:
                new_start = 128//16 - sampled_mask[:,3]
                new_end =  128//16 - sampled_mask[:,2]
                sampled_mask[:,2] = new_start
                sampled_mask[:,3] = new_end
            
            if self.new_eval:
                return imgs,pid,ambi,cam,torch.tensor(sampled_mask,dtype=torch.int16)
            else:
                return imgs,pid,cam,torch.tensor(sampled_mask,dtype=torch.int16)
        else:
            if self.new_eval:
                raise NotImplementedError
            else:
                return imgs,pid,cam




