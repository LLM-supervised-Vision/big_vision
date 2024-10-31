# !/bin/bash

export TFDS_DATA_DIR=gs://us-central2-storage/tensorflow_datasets

cd ~/austin_big_vision
# cambrian_dataset, generative
python -m big_vision.trainers.proj.paligemma.train \
    --config big_vision/configs/proj/paligemma/paligemma_unified_configs.py:dataset_name='cambrian_dataset/10M:1.0.0',epoch=10.0,debug=True,debug_tiny_model=True \

# python -m big_vision.trainers.proj.paligemma.train \
#     --config big_vision/configs/proj/paligemma/paligemma_playground.py:res=224,mode=generative,loss_fn=softmax,dataset_name=datacomp_recap/10M:1.0.0,org_caption_ratio=1.0,datacomp_backbone=gemma_supervised,drop_path_rate=0.0,lr=1e-3,wd=1e-4,epoch=10.0,freeze_vit=False,img_variant=B/16,img_beit_init=False,img_qknorm=False,freeze_llm=False,llm_variant=gemma_2b,llm_ckpt=partial_frozen:9,llm_head=gap,llm_lr_mult=0.01,llm_dropout=0.0,llm_clean_vocab=True,llm_projection=True,llm_text_len=128,batch_size=16384,total_samples=3.0,dtype=bfloat16,debug=True \
#     --workdir gs://us-central2-storage/tensorflow_datasets/mllm_ckpts/paligemma/gemma2b-partial_frozen99-0.01-drop0.0-vocab256128-gap-projT-txtlen128_b16-beitF-qknormF_generative_datacomp-org_ratio1.0-10M-epoch10_dpr0.0_bs16k_s3b_lr1e-3_wd1e-4_bf16_10-29_1338:params
#     # --config big_vision/configs/proj/paligemma/paligemma_playground.py:res=224,mode='generative',loss_fn='softmax',dataset_name='cambrian_dataset/10M:1.0.0',org_caption_ratio=0.5,datacomp_backbone='gemma_supervised',drop_path_rate=0.0,lr=1e-3,wd=1e-4,epoch=10.0,freeze_vit=False,img_variant='B/16',img_beit_init=False,img_qknorm=False,freeze_llm=False,llm_variant='gemma_2b',llm_ckpt='partial_frozen:9',llm_head='gap',llm_lr_mult=0.1,llm_dropout=0.0,llm_clean_vocab=True,llm_projection=True,batch_size=16384,total_samples=3.0,dtype='bfloat16',debug=True \
#     # # --config big_vision/configs/proj/paligemma/paligemma_playground.py:res=224,mode='contrastive',loss_fn='softmax',dataset_name='laion400m/images',datacomp_inkey='re_caption',datacomp_backbone='gemma_supervised',drop_path_rate=0.0,lr=1e-3,wd=1e-4,epoch=5.0,freeze_vit=False,img_variant='B/16',img_beit_init=False,img_qknorm=False,freeze_llm=False,llm_variant='gemma_2b',llm_ckpt='adapter',llm_head='ffn',llm_lr_mult=1.0,llm_dropout=0.0,llm_clean_vocab=True,llm_projection=False,batch_size=16384,total_samples=3.0,dtype='bfloat16',debug=True \

