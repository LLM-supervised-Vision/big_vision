# !/bin/bash

export TFDS_DATA_DIR=gs://us-central2-storage/tensorflow_datasets

cd ~/austin_big_vision
python -m big_vision.trainers.proj.image_text.siglip \
    --config big_vision/configs/proj/image_text/clip_exact.py:loss_fn='softmax',unified=True,lit=False,memory_efficient=False,debug=True \
    # --workdir gs://us-central2-storage/tensorflow_datasets/lit_ckpts/clip_bs16384_warm10k_lr1e-3_wd1e-4_bf16_qknorm-F_b2-0.95_12lyr_07-23_1510
    # --config big_vision/configs/proj/image_text/clip_exact.py:loss_fn='sigmoid',unified=False,lit=True,memory_efficient=False,debug=True # lit
    # --config big_vision/configs/proj/image_text/siglip_lit_cc12m.py \
    # --workdir gs://us-central2-storage/tensorflow_datasets/siglip_lit_cc12m_`date '+%m-%d_%H%M'`
    # --config big_vision/configs/proj/image_text/siglip_lit_coco.py


    # --config big_vision/configs/proj/image_text/siglip_replication.py:loss_fn='sigmoid',autoregressive=False,batch_size=8192,scan=True,fsdp=2,dtype='bfloat16',debug=True \
    # --workdir gs://us-central2-storage/tensorflow_datasets/siglip_replication_bs8192_`date '+%m-%d_%H%M'`
    # --workdir gs://us-central2-storage/tensorflow_datasets/siglip_replication_pod_04-11_2247

    # --config big_vision/configs/proj/image_text/siglip_replication.py:loss_fn='softmax',autoregressive=True,batch_size=8192,scan=True,fsdp=2,dtype='bfloat16',debug=False \
    # --workdir gs://us-central2-storage/tensorflow_datasets/clip-autoregressive_replication_bs8192_`date '+%m-%d_%H%M'`