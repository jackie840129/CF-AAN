# for testing mode.
# (1) TEST.TEST_MODE 'test' (using RRS to sample)
# (2) TEST.TEST_MODE 'test_0' (first T images)
# (3) TEST.TEST_MODE 'test_all_sampled' (using RRS to sample T,average the N/T tracklets)
# (4) TEST.TEST_MODE 'test_all_continuous' (continuous smaple T frames, average the N/T tracklets)

python3 tools/test.py --config_file='configs/video_baseline.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('mars',)"    MODEL.NON_LAYERS [0,2,3,0] \
                 MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('/home/xxxx/xxxx.pth')" \
                 MODEL.NAME 'resnet50_axial' INPUT.SEQ_LEN 6 MODEL.TEMP 'Done' TEST.TEST_MODE 'test_all_sampled' TEST.IMS_PER_BATCH 1  \
                 DATASETS.ROOT_DIR '/work/sychien421/Dataset/' TEST.NEW_EVAL False
