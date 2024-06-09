# !/bin/bash

export TFDS_DATA_DIR=gs://us-central2-storage/tensorflow_datasets

cd ~/austin_big_vision
python -m big_vision.trainers.proj.image_text.siglip \
    --config big_vision/configs/proj/image_text/siglip_replication.py:loss_fn='softmax',autoregressive=True,batch_size=8192,scan=True,fsdp=2,dtype='bfloat16',debug=False \
    --workdir gs://us-central2-storage/tensorflow_datasets/clip-autoregressive_replication_bs8192_`date '+%m-%d_%H%M'`

    # --config big_vision/configs/proj/image_text/siglip_replication.py:loss_fn='sigmoid',autoregressive=False,batch_size=8192,scan=True,fsdp=2,dtype='bfloat16',debug=False \
    # --workdir gs://us-central2-storage/tensorflow_datasets/siglip_replication_bs8192_`date '+%m-%d_%H%M'`

    # --config big_vision/configs/proj/image_text/siglip_lit_laion400m.py \
    # --workdir gs://us-central2-storage/tensorflow_datasets/siglip_lit_laion400m_`date '+%m-%d_%H%M'`
    # --config big_vision/configs/proj/image_text/siglip_lit_coco.py