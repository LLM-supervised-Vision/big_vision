# !/bin/bash

export TFDS_DATA_DIR=gs://us-central2-storage/tensorflow_datasets

cd ~/austin_big_vision
python -m big_vision.trainers.proj.coca.coca \
    --config big_vision/configs/proj/coca/coca_replication.py:batch_size=4096,lr=1e-4,dtype='float32',eval_only=False,debug=False \
    --workdir gs://us-central2-storage/tensorflow_datasets/ckpts/coca_replication_bs4096_1.0co-1.0ca_lr1e-4_fp32_`date '+%m-%d_%H%M'`
