# !/bin/bash

export TFDS_DATA_DIR=gs://us-central2-storage/tensorflow_datasets

cd ~/austin_big_vision
python -m big_vision.trainers.proj.coca.coca \
    --config big_vision/configs/proj/coca/coca_replication.py:batch_size=512,total_steps=5859375,eval_only=False \
    --workdir gs://us-central2-storage/tensorflow_datasets/ckpts/coca_replication_bs512_1.0co-2.0ca_lr5e-4_`date '+%m-%d_%H%M'`
