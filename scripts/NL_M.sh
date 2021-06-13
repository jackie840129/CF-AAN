python3 tools/train.py --config_file='configs/video_baseline.yml' MODEL.DEVICE_ID "('1,2')" DATASETS.NAMES "('mars',)" \
                                                                   OUTPUT_DIR "./ckpt_DL_M/MARS_DL_s6_NL0230" SOLVER.SOFT_MARGIN True \
                                                                   MODEL.NON_LAYERS [0,2,3,0] INPUT.IF_RE True INPUT.IF_CROP False MODEL.IF_LABELSMOOTH 'no' \
                                                                   MODEL.NAME 'resnet50_NL' INPUT.SEQ_LEN 6 MODEL.TEMP 'Done' \
                                                                   DATASETS.ROOT_DIR '/home/mediax/Dataset/'
