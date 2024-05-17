# !/bin/bash

export TFDS_DATA_DIR=gs://us-central2-storage/tensorflow_datasets

cd ~/austin_big_vision
python -m big_vision.trainers.proj.cappa.generative \
  --config big_vision/configs/proj/cappa/cappa_replication.py:total_steps=366_500
  # --config big_vision/configs/proj/cappa/pretrain.py
  # --workdir gs://us-central2-storage/tensorflow_datasets/cappa_replication_`date '+%m-%d_%H%M'`
