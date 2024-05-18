# !/bin/bash

export TFDS_DATA_DIR=gs://us-central2-storage/tensorflow_datasets

cd ~/austin_big_vision
python -m big_vision.trainers.proj.coca.coca \
    --config big_vision/configs/proj/coca/coca_replication.py
    # --workdir gs://us-central2-storage/tensorflow_datasets/coca_replication_`date '+%m-%d_%H%M'`
