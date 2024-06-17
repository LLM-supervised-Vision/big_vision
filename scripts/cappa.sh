# !/bin/bash

export TFDS_DATA_DIR=gs://us-central2-storage/tensorflow_datasets

cd ~/austin_big_vision
python -m big_vision.trainers.proj.cappa.generative \
  --config big_vision/configs/proj/cappa/cappa_replication.py:batch_size=16384,total_steps=183_105,eval_only=False \
  --workdir gs://us-central2-storage/tensorflow_datasets/cappa_replication_`date '+%m-%d_%H%M'`
  # --config big_vision/configs/proj/cappa/pretrain.py
