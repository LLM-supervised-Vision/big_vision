# !/bin/bash

export TFDS_DATA_DIR=gs://us-central2-storage/tensorflow_datasets

cd ~/austin_big_vision
python -m big_vision.trainers.proj.paligemma.train \
    --config big_vision/configs/proj/paligemma/paligemma_playground.py:res=224,mode='contrastive',loss_fn='softmax',dataset_name='datacomp_recap/50M:1.0.0',datacomp_inkey='re_caption',datacomp_backbone='gemma_supervised',drop_path_rate=0.0,lr=1e-3,wd=1e-4,epoch=10.0,freeze_vit=False,img_variant='B/16',img_beit_init=False,img_qknorm=False,freeze_llm=False,llm_variant='gemma_2b',llm_ckpt='partial_frozen:9',llm_head='gap',llm_lr_mult=0.1,llm_dropout=0.0,llm_clean_vocab=True,llm_projection=True,batch_size=16384,total_samples=3.0,dtype='bfloat16',debug=True \
    # --config big_vision/configs/proj/paligemma/paligemma_playground.py:res=224,mode='contrastive',loss_fn='softmax',dataset_name='laion400m/images',datacomp_inkey='re_caption',datacomp_backbone='gemma_supervised',drop_path_rate=0.0,lr=1e-3,wd=1e-4,epoch=5.0,freeze_vit=False,img_variant='B/16',img_beit_init=False,img_qknorm=False,freeze_llm=False,llm_variant='gemma_2b',llm_ckpt='adapter',llm_head='ffn',llm_lr_mult=1.0,llm_dropout=0.0,llm_clean_vocab=True,llm_projection=False,batch_size=16384,total_samples=3.0,dtype='bfloat16',debug=True \
