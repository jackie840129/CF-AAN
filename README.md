# Video-based Person Re-identification without Bells and Whistles

[[Paper]](http://media.ee.ntu.edu.tw/research/CFAAN/paper/CVPRw21_VideoReID.pdf) [[arXiv]](https://arxiv.org/pdf/2105.10678.pdf) [[video]](https://youtu.be/RNssJNmq504)

[Chih-Ting Liu](https://jackie840129.github.io/), [Jun-Cheng Chen](https://www.citi.sinica.edu.tw/pages/pullpull/contact_en.html), [Chu-Song Chen](https://imp.iis.sinica.edu.tw/) and [Shao-Yi Chien](http://www.ee.ntu.edu.tw/profile?id=101),<br/>Analysis & Modeling of Faces & Gestures Workshop jointly with IEEE Conference on Computer Vision and Pattern Recognition (**CVPRw**), 2021

This is the pytorch implementatin of Coarse-to-Fine Axial Attention Network **(CF-AAN)** for video-based person Re-ID. 
<br/>It achieves **91.3%** in rank-1 accuracy and **86.5%** in mAP on our aligned MARS dataset.

## News

**`2021-06-13`**: 
- We release the code and aligned dataset for our work.
- We update the Readme related to our new dataset, and the others will be updated gradually.

**`2021-06-18`**:
- We update the description for training and testing CF-AAN.

## Aligned dataset with our re-Detect and Link module

### Download Link : 

- MARS (DL) : [[Google Drive]](https://drive.google.com/file/d/1adP39y7xoKYX8Z4lyBtZiDTg9kZyK1Cx/view?usp=sharing)
- For DukeV, we didn't perform DL on DukeMTMC-VideoReID because the bounding boxes are greound truth annotations.

### Results
The video tracklet will be re-Detected, linked (tracking) and padded to the original image size, as follow.
<p align="left"><img src='imgs/DL.png' width="310pix"></p>

### Folder Structure
MARS dataset:
```
MARS-DL/
|-- bbox_train/
|-- bbox_test/
|-- info/
|-- |-- mask_info.csv (for DL mask)
|-- |-- mask_info_test.csv  (for DL mask)
|-- |-- clean_tracks_test_info.mat (for new evaluation protocol)
|-- |-- .... (other original info files)
```
DukeV dataset:
```
DukeMTMC-VideoReID/
|-- train/
|-- gallery/
|-- query/
```
You can put this two folders under your root dataset directory.
```
path to your root dir/
|-- MARS-DL/
|-- DukeMTMC-VideoReID/
```
## Coarse-to-Fine Axial Attention Network (CF-AAN)

### Requirement
We use Python 3.6, Pytorch 1.5 and Pytorch-ignite in this project. To install required modules, run:
```
pip3 install -r requirements.txt
```
### Training
#### Train CF-AAN on MARS-DL
You can alter the argument in `scripts/AA_M.sh` and run it with:
```
sh scripts/AA_M.sh
```
Or, you can directly type:
```
python3 tools/train.py --config_file='configs/video_baseline.yml' MODEL.DEVICE_ID "('0,1')" DATASETS.NAMES "('mars',)" INPUT.SEQ_LEN 6 \
                                                                   OUTPUT_DIR "./ckpt_DL_M/MARS_DL_s6_resnet_axial_gap_rqkv_gran4" SOLVER.SOFT_MARGIN True \
                                                                   MODEL.NAME 'resnet50_axial' MODEL.TEMP 'Done' INPUT.IF_RE True \
                                                                   DATASETS.ROOT_DIR '<PATH TO DATASET ROOT DIRECTORY>'
```
\* `<PATH TO DATASET ROOT DIRECTORY>` is the directory containing both MARS and DukeV dataset.
#### Train Non-local or baseline on MARS
You can alter the argument in `scripts/NL_M.sh` & `scripts/baseline_M.sh` and run it with:

`sh scripts/AA_M.sh` & `sh scripts/baseline_M.sh`
#### Train models on DukeMTMC-VideoReID
You can use the scripts `scripts/AA_D.sh`, `scripts/NL_D.sh`, & `scripts/baseline_D.sh`

#### Notes
If you want to train on original MARS dataset, you just need to change the comment in `data/datasets/MARS.py` :
```
class MARS(BaseVideoDataset):
    dataset_dir = 'MARS'
    # dataset_dir = 'MARS-DL'
    info_dir = 'info
```

### Testing
You can alter the argument in `scripts/test_M.sh` and run it with:
```
sh scripts/test_M.sh
```
\* `TEST.WEIGHT` is the path for the saved pytorch (.pth) model.

\* There are four modes for `TEST.TEST_MODE`.
1. `TEST.TEST_MODE 'test'` 
    * Use RRS[3] testing mode, which samples the first image of T snippets split from tracklet.
2. `TEST.TEST_MODE 'test_0'` 
    * Sample first T images in tracklet.
3. `TEST.TEST_MODE 'test_all_sampled'`
    * Create N/T tracklets (all 1st image from T RRS snippets, all 2nd from T RRS snippets...), and average the N/T features. 
4. `TEST.TEST_MODE 'test_all_continuous'` 
    * Continuous smaple T frames, create N/T tracklets, and average the N/T features.

If you want to test on DukeV, you can  just alter the corresponding arguments in `scripts/test_M.sh`.

## New Evaluatoin Protocol

Change the `TEST.NEW_EVAL False` to `TEST.NEW_EVAL True`.

The details will be introduced soon.

## Citation
```
@InProceedings{Liu_2021_CVPR,
    author    = {Liu, Chih-Ting and Chen, Jun-Cheng and Chen, Chu-Song and Chien, Shao-Yi},
    title     = {Video-Based Person Re-Identification Without Bells and Whistles},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {1491-1500}
}
```
## Reference

1. The structure of our code are based on [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline).  
2. Some codes of our CF-AAN are based on [axial-deeplab](https://github.com/csrhddlam/axial-deeplab)
3. Li, Shuang, et al. "Diversity regularized spatiotemporal attention for video-based person re-identification." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
## Contact

[Chih-Ting Liu](https://jackie840129.github.io/), [Media IC & System Lab](https://github.com/mediaic), National Taiwan University

E-mail : jackieliu@media.ee.ntu.edu.tw
