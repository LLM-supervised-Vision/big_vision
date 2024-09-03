# !/bin/bash


cd ~/austin_big_vision
export PYTHONPATH=$PYTHONPATH:/home/austinwang/austin_big_vision
python /home/austinwang/austin_big_vision/big_vision/trainers/proj/paligemma/run_llm.py \
    --ckpt /home/austinwang/gemma2b.npz
    # --ckpt gs://us-central2-storage/tensorflow_datasets/mllm_ckpts/paligemma/gemma2b-partial_frozen99-0.01-gap_b16-F_contrastive_bs16k_s3b_lr1e-3_wd1e-4_bf16_09-01_0446/checkpoint.bv-000183105:llm