python3 tools/train.py --config_file='configs/video_baseline.yml' MODEL.DEVICE_ID "('0,1')" DATASETS.NAMES "('mars',)" \
                                                                   OUTPUT_DIR "./ckpt_DL_M/MARS_DL_avgpool_s6" SOLVER.SOFT_MARGIN True \
                                                                   MODEL.IF_LABELSMOOTH 'no'  INPUT.IF_RE True INPUT.SEQ_LEN 6 TEST.NEW_EVAL False \
                                                                   DATASETS.ROOT_DIR '/home/mediax/Dataset/'
