python3 tools/train.py --config_file='configs/video_baseline.yml' MODEL.DEVICE_ID "('0,1')" DATASETS.NAMES "('dukev',)" INPUT.SEQ_LEN 6 \
                                                                   OUTPUT_DIR "./ckpt_DL_duke/Duke_DL_s6_resnet_axial_gap_sine_gran2" SOLVER.SOFT_MARGIN True \
                                                                   MODEL.NAME 'resnet50_axial' MODEL.TEMP 'Done' MODEL.IF_LABELSMOOTH 'no'  INPUT.IF_RE True \
                                                                   DATASETS.ROOT_DIR '/home/mediax/Dataset/' TEST.NEW_EVAL False
