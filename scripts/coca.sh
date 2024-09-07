# !/bin/bash

export TFDS_DATA_DIR=gs://us-central2-storage/tensorflow_datasets

cd ~/austin_big_vision
python -m big_vision.trainers.proj.coca.coca \
    --config big_vision/configs/proj/coca/coca_replication.py:batch_size=8192,warmup_ratio=0.02,lr=1e-3,dtype='bfloat16',dec_lyr=6,masked_pred_prob=0.75,scan=False,eval_only=False,debug=True \
    # --workdir gs://us-central2-storage/tensorflow_datasets/ckpts/coca_replication_bs8192_warm0.02_1.0co-1.0ca_lr1e-3_bf16_b2-0.95_3lyr_`date '+%m-%d_%H%M'`
    # --workdir gs://us-central2-storage/tensorflow_datasets/ckpts/coca_replication_bs8192_warm0.02_1.0co-1.0ca_lr1e-3_bf16_b2-0.95_3lyr_06-08_0358
