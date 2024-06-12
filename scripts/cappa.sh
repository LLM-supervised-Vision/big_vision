# !/bin/bash

export TFDS_DATA_DIR=gs://us-central2-storage/tensorflow_datasets

cd ~/austin_big_vision
python -m big_vision.trainers.proj.cappa.generative \
  --config big_vision/configs/proj/cappa/cappa_replication.py:batch_size=4096,total_steps=91_553,eval_only=True \
  # --workdir gs://us-central2-storage/tensorflow_datasets/cappa_replication_`date '+%m-%d_%H%M'`
  # --config big_vision/configs/proj/cappa/pretrain.py
