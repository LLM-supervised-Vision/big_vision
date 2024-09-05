# !/bin/bash

export TFDS_DATA_DIR=gs://us-central2-storage/tensorflow_datasets

cd ~/austin_big_vision
python -m big_vision.trainers.proj.paligemma.train \
    --config big_vision/configs/proj/paligemma/paligemma_playground.py:res=224,mode='contrastive',freeze_vit=False,freeze_llm=False,llm_ckpt='partial_frozen:9',llm_pool='eos',llm_dropout=0.0,llm_projection=True,batch_size=256,dtype='bfloat16',total_samples=3.0,debug=True
