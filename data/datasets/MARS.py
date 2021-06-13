# encoding: utf-8
import glob
import re

import os.path as osp
from scipy.io import loadmat
from .bases import BaseVideoDataset
import pandas as pd


class MARS(BaseVideoDataset):
    # dataset_dir = 'MARS'
    dataset_dir = 'MARS-DL'
    info_dir = 'info'

    def __init__(self, root='/home/mediax/Dataset', verbose=True, min_seq_len =0,new_eval=False):
        super(MARS, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.info_dir = osp.join(self.dataset_dir,self.info_dir)
        self.train_name_path = osp.join(self.info_dir,'train_name.txt')
        self.test_name_path = osp.join(self.info_dir,'test_name.txt')
        self.track_train_info_path = osp.join(self.info_dir,'tracks_train_info.mat')
        self.track_test_info_path = osp.join(self.info_dir,'tracks_test_info.mat')
        self.query_IDX_path = osp.join(self.info_dir,'query_IDX.mat')
        self.new_eval = new_eval
        if self.new_eval:
            self.track_test_info_path = osp.join(self.info_dir,'clean_tracks_test_info.mat')

        if 'DL' in self.dataset_dir:
            train_mask_csv = pd.read_csv(osp.join(self.info_dir,'mask_info.csv'),sep=',',header=None).values
            test_mask_csv = pd.read_csv(osp.join(self.info_dir,'mask_info_test.csv'),sep=',',header=None).values
        else:
            train_mask_csv,test_mask_csv = None,None
        self._check_before_run()
        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)
        track_train = loadmat(self.track_train_info_path)['track_train_info'] #(8298,4) 
        track_test = loadmat(self.track_test_info_path)['track_test_info'] #(12180,4)

        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze()-1  #(1980,) start from 0
        track_query = track_test[query_IDX,:]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX,:]
        # track_gallery = track_test

        train = self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True,min_seq_len=min_seq_len,mask_info=train_mask_csv)
        query = self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False,mask_info=test_mask_csv,new_eval=self.new_eval)
        gallery = self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False,mask_info=test_mask_csv,new_eval=self.new_eval)

        if verbose:
            print("=> MARS loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train # list of tuple--(paths,id,cams)
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_tracklets, self.num_train_cams = self.get_videodata_info(self.train)
        self.num_query_pids, self.num_query_tracklets, self.num_query_cams = self.get_videodata_info(self.query)
        self.num_gallery_pids, self.num_gallery_tracklets, self.num_gallery_cams = self.get_videodata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self,names, meta_data, home_dir=None,relabel=False,min_seq_len=0,mask_info=None,new_eval=False):
        assert home_dir in ['bbox_train','bbox_test']

        n_tracklets = len(meta_data)
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        
        tracklets = []
        num_imgs_per_tracklet = []
        for tracklet_idx in range(n_tracklets):
            data = meta_data[tracklet_idx,...]
            if new_eval == True:
                start_idx,end_idx,pid,cam, new_pid, new_ambi = data
            else:
                start_idx,end_idx,pid,cam = data
            if pid == -1 or pid == 0 : continue  # junk index
            assert 1<= cam <=6
            
            if relabel : pid = pid2label[pid]
            cam -= 1
            img_names = names[start_idx-1:end_idx]

            if mask_info is not None:
                masks = mask_info[start_idx-1:end_idx,1:].astype('int16')//16

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.dataset_dir,home_dir,img_name[:4],img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                if mask_info is not None:
                    masks = mask_info[start_idx-1:end_idx,1:].astype('int16')//16
                    if new_eval == True:
                        tracklets.append((img_paths,pid,new_pid,new_ambi,cam, masks))
                    else:
                        tracklets.append((img_paths,pid,cam,masks))
                else:
                    if new_eval == True:
                        tracklets.append((img_paths,pid,new_pid,new_ambi,cam))
                    else:
                        tracklets.append((img_paths,pid,cam))
                # num_imgs_per_tracklet.append(len(img_paths))
        # n_tracklets = len(tracklets)

        return tracklets
