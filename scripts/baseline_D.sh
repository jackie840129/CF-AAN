python3 tools/train.py --config_file='configs/video_baseline.yml' MODEL.DEVICE_ID "('0,1')" DATASETS.NAMES "('dukev',)" \
                                                                   OUTPUT_DIR "./ckpt_DL_duke/Duke_avgpool" SOLVER.SOFT_MARGIN True \
                                                                   MODEL.IF_LABELSMOOTH 'no'  INPUT.IF_RE True INPUT.SEQ_LEN 6 \
                                                                   DATASETS.ROOT_DIR '/work/sychien421/Dataset/'
