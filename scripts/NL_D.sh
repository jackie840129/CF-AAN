python3 tools/train.py --config_file='configs/video_baseline.yml' MODEL.DEVICE_ID "('0,1')" DATASETS.NAMES "('dukev',)" \
                                                                   OUTPUT_DIR "./ckpt_DL_duke/Duke_DL_s6_NL0230_2gpu" SOLVER.SOFT_MARGIN True \
                                                                   MODEL.NON_LAYERS [0,2,3,0] INPUT.IF_RE True INPUT.IF_CROP False MODEL.IF_LABELSMOOTH 'no' \
                                                                   MODEL.NAME 'resnet50_NL' INPUT.SEQ_LEN 6 MODEL.TEMP 'Done' \
                                                                   DATASETS.ROOT_DIR '/work/sychien421/Dataset/'
