# !/bin/bash

export TFDS_DATA_DIR=gs://us-central2-storage/tensorflow_datasets

cd ~/austin_big_vision
python -m big_vision.trainers.proj.paligemma.train \
    --config big_vision/configs/proj/paligemma/pailgemma_playground.py:res=224,mode='contrastive',freeze_vit=False,freeze_llm=True,batch_size=256,total_samples=3.0,debug=True
