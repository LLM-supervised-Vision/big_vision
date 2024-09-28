# !/bin/bash

export TFDS_DATA_DIR=gs://us-central2-storage/tensorflow_datasets

cd ~/austin_big_vision
python -m big_vision.trainers.proj.image_text.siglip \
    --config big_vision/configs/proj/image_text/clip_exact.py:loss_fn='softmax',unified=True,lit=False,dataset_name='datacomp_recap/10M:1.0.0',scale='gemma',memory_efficient=True,debug=True