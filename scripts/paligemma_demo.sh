# !/bin/bash


cd ~/austin_big_vision
export PYTHONPATH=$PYTHONPATH:/home/austinwang/austin_big_vision
python /home/austinwang/austin_big_vision/big_vision/trainers/proj/paligemma/run.py \
    --ckpt /home/austinwang/gemma2b.npz