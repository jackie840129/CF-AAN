# encoding: utf-8
import glob
import re
import json
import pickle
import os
import os.path as osp
from scipy.io import loadmat
from .bases import BaseVideoDataset
import pandas as pd
import numpy as np


class DukeV(BaseVideoDataset):
    dataset_dir = 'DukeMTMC-VideoReID'

    def __init__(self, root='/home/mediax/Dataset', verbose=True, min_seq_len =0,info_dir='./DukeV_info',new_eval=False):
        super(DukeV, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir,'train')
        self.gallery_dir = osp.join(self.dataset_dir,'gallery')
        self.query_dir = osp.join(self.dataset_dir,'query')
        self.min_seq_len = min_seq_len
        #for self-created duke info
        if 'DL' in self.dataset_dir:
            info_dir = './DukeV_DL_info'
        self.train_pkl = osp.join(info_dir,'train.pkl')
        self.gallery_pkl = osp.join(info_dir,'gallery.pkl')
        self.query_pkl = osp.join(info_dir,'query.pkl')
        self.info_dir = info_dir
        self._check_before_run()

        if 'DL' in self.dataset_dir:
            train_mask_csv = pd.read_csv(osp.join(self.dataset_dir,'duke_mask_info.csv'),sep=',',header=None).values
            query_mask_csv = pd.read_csv(osp.join(self.dataset_dir,'duke_mask_info_query.csv'),sep=',',header=None).values
            gallery_mask_csv = pd.read_csv(osp.join(self.dataset_dir,'duke_mask_info_gallery.csv'),sep=',',header=None).values
        else:
            train_mask_csv,query_mask_csv, gallery_mask_csv = None,None,None

        train = self._process_dir(self.train_dir,self.train_pkl,relabel=True,mask_info=train_mask_csv)
        gallery = self._process_dir(self.gallery_dir,self.gallery_pkl,relabel=False,mask_info=gallery_mask_csv)
        query = self._process_dir(self.query_dir,self.query_pkl,relabel=False,mask_info=query_mask_csv)
        if verbose:
            print("=> DukeV loaded")
            self.print_dataset_statistics(train, query, gallery)
        self.train = train # list of tuple--(paths,id,cams)
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_tracklets, self.num_train_cams = self.get_videodata_info(self.train)
        self.num_query_pids, self.num_query_tracklets, self.num_query_cams = self.get_videodata_info(self.query)
        self.num_gallery_pids, self.num_gallery_tracklets, self.num_gallery_cams = self.get_videodata_info(self.gallery)

    def _process_dir(self,dir_path,pkl_path,relabel,mask_info=None):

        if osp.exists(pkl_path):
            print('==> %s exisit. Load...'%(pkl_path))
            with open(pkl_path,'rb') as f:
                pkl_file = pickle.load(f)
            
            if mask_info is None:
                return pkl_file

            tracklets = []
            start = 0
            for info in pkl_file:
                end = start + len(info[0])
                tracklets.append((info[0],info[1],info[2],mask_info[start:end,1:].astype('int16')//16))
                start = end
            return tracklets

        pdirs = sorted(glob.glob(osp.join(dir_path, '*')))
        print("Processing {} with {} person identities".format(dir_path, len(pdirs)))
        pids = sorted(list(set([int(osp.basename(pdir)) for pdir in pdirs])))
        pid2label = {pid : label for label,pid in enumerate(pids)}

        tracklets = []
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            if relabel : pid = pid2label[pid]
            track_dirs = sorted(glob.glob(osp.join(pdir,'*')))
            for track_dir in track_dirs:
                img_paths = sorted(glob.glob(osp.join(track_dir,'*.jpg')))
                num_imgs = len(img_paths)
                if num_imgs < self.min_seq_len :
                    continue
                img_name = osp.basename(img_paths[0])
                if img_name.find('_') == -1 :
                    camid = int(img_name[5])-1
                else:
                    camid = int(img_name[6])-1
                img_paths = tuple(img_paths)
                tracklets.append((img_paths,pid,camid))
        # save to pickle
        if not osp.isdir(self.info_dir):
            os.mkdir(self.info_dir)
        with open(pkl_path,'wb') as f:
            pickle.dump(tracklets,f)
        return tracklets

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
