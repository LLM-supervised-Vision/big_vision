import os
import logging
import json
import tensorflow as tf
import ml_collections
from ml_collections import ConfigDict

import big_vision.configs.common as bvcc
from big_vision.configs.proj.image_text import common
from big_vision.configs.proj.paligemma.transfers.common import combine_and_keep_train, combine_and_keep_eval, TOKENIZER

def training_data(dataset_name, caption_key, *, res=224, prefix='', text_len=64):
  assert dataset_name.split('/')[0] in ['laion400m', 'datacomp_recap'], f'Unknown dataset_name: {dataset_name}'
  c = bvcc.parse_arg('') 
  c.data = dict(
    name=dataset_name,
    split='train',
    data_dir='gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets'
  )
  c.pp = '|'.join([
    f'decode|resize({res})|value_range(-1,1)',
    f'strfmt("{prefix}", outkey="prefix")',
    f'copy(inkey="{caption_key}", outkey="suffix")',
    combine_and_keep_train(text_len),
  ])
  return c

def add_eval(c, res, *, text_len=64, prefix, mode, **kw):
  if mode == "contrastive":
    c.evals.retrieval_coco = common.get_coco(
      pred='contrastive_logits',
      pp_img=f'resize({res})|value_range(-1, 1)',
      pp_txt='|'.join([
        f'strfmt("{prefix}", outkey="prefix")',
        'copy(inkey="texts", outkey="suffix")',
        combine_and_keep_eval(text_len,eos='yes'),
        f'copy(inkey="text", outkey="labels")',
      ]),
      log_steps=1000,
    )
    c.evals.retrieval_coco.update(kw)

    c.evals.zeroshot_imagenet = common.get_disclf(
      pred='contrastive_logits',
      sz=res,
      pp_txt='|'.join([
        f'strfmt("{prefix}", outkey="prefix")',
        'copy(inkey="texts", outkey="suffix")',
        combine_and_keep_eval(text_len,eos='yes'),
        f'copy(inkey="text", outkey="labels")',
      ]),
      dataset_names=('imagenet2012','imagenet_v2','imagenet2012_real'),
      log_steps=1000,
    )
    c.evals.zeroshot_imagenet.update(kw)

  elif mode == "generative":
    c.evals = {}
  #   pp = '|'.join([
  #       f'strfmt("{prefix}", outkey="prefix")',
  #       'copy(inkey="label", outkey="suffix")',
  #       combine_and_keep_eval(text_len, keep=('text', 'mask_ar')),
  #       f'copy(inkey="text", outkey="labels")',
  #   ])
  #   c.evals['imagenet/scoring'] = dict(
  #     type='proj.cappa.scoring_classifier',
  #     pred='score',
  #     log_percent=0.1,
  #     data=dict(name='imagenet2012', split='validation'),
  #     pp_fn=f'decode|resize({res})|keep("image", "label")',
  #     pp_txt=pp,
  #   )
  else:
    raise ValueError(f"Unknown mode: {mode}")

backbone_dict = {
  'gemma_2b': 'gs://us-central2-storage/tensorflow_datasets/gemma2b.npz',
  'clip': 'gs://us-central2-storage/tensorflow_datasets/vit-b-16_3b_pretraining/clip_bs16384_warm0.03_lr1e-3_wd1e-4_bf16_qknorm-F_b2-0.95_12lyr_07-25_1415',
  'gemma_supervised': 'gs://us-central2-storage/tensorflow_datasets/mllm_ckpts/paligemma/gemma2b-partial_frozen99-0.01-gap_b16-F_contrastive_bs16k_s3b_lr1e-3_wd1e-4_bf16_09-01_0446',
}

def get_config(arg=None):
  c = bvcc.parse_arg(
    arg,
    objective='contrastive', loss_fn='softmax', lr=1e-3, wd=1e-4, batch_size=16384, total_epochs=-1, total_samples=-1.0,
    training_mode='pretrain', res=224, dataset_name='laion400m/images', caption_key='caption',
    img_variant='B/16', img_trainable='scratch', img_beit_init=False, img_qknorm=False,
    llm_variant='gemma_2b', llm_trainable='full', llm_head='none', llm_lr_mult=0.1, llm_dropout=0.0, llm_clean_vocab=False, llm_projection=False, llm_text_len=64, 
    drop_path_rate=0.0, dtype='float32',
    debug=False,
	)
  
	# Input Data Section
  c.input = training_data(dataset_name=c.dataset_name, caption_key=c.caption_key, res=c.res, prefix='', text_len=c.llm_text_len)
  c.input.batch_size = c.batch_size
  if c.total_epochs > 0:
    assert c.total_samples < 0, "total_epochs and total_samples cannot be specified together"
    c.total_steps = c.total_epochs * c.input.total_samples // c.batch_size
  elif c.total_samples > 0:
    c.total_steps = c.total_samples // c.batch_size
  else:
    raise ValueError("Either total_epochs or total_samples must be specified")
  
  # Model Config Section
  c.model_name = 'proj.paligemma.paligemma'
  c.model = dict(
    temperature_init = 1/0.07 if c.loss_fn == 'softmax' else 10.0,
    bias_init = None if c.loss_fn == 'softmax' else -10.0,
  )
  c.model.img = dict(
    variant=c.img_variant,
    posemb='learn', rep_size=False, dropout=0.0, pool_type='none', 
    head_zeroinit=False, beit_init=c.img_beit_init, mask=None, normalize_qk=c.img_qknorm, scan=True, 
    remat_policy='nothing_saveable', dtype_mm=c.dtype, proj_bias=False, drop_path_rate=c.drop_path_rate,
  )
  c.model.llm = dict(
    variant=c.llm_variant, 
    scan=True, remat_policy='nothing_saveable', vocab_size=None, dropout=c.llm_dropout, 
    dropout_bdims=(), cache_dtype=None, dtype=c.dtype, lyrs_frozen=-1, head=c.llm_head, 
    projection=c.llm_projection, proj_bias=False, drop_path_rate=c.drop_path_rate,
  )
  if not c.llm_clean_vocab: c.model.llm['vocab_size'] = 256_000 + 1024 + 128
  if c.loss_fn == 'sigmoid': c.model.img['proj_bias'], c.model.llm['proj_bias'] = True, True
  if 'partial_frozen:' in c.llm_trainable: 
    c.llm_trainable, lyrs_frozen = c.llm_ckpt.split(':')
    c.model.llm['lyrs_frozen'] = int(lyrs_frozen)

  # Model Initialization Section
  c.model_init = {'img': None, 'llm': None}
  if c.training_mode.split('_')[0] == 'pretrain':
    # model = scratch vit + pre-trained gemma_2b
    llm_backbone = backbone_dict[c.llm_variant]
    c.model_init['llm'] = llm_backbone
  elif c.training_mode.split('_')[0] == 'finetune': 
    assert c.training_mode.split('_')[1] in ['clip+llm','gemma_supervised'], f'Unknown training_mode: {c.training_mode}'
    backbone = backbone_dict[c.training_mode.split('_')[1]]
    ckpt_cfg_path = f'{backbone}/config.json'
    ckpt_cfg = ml_collections.ConfigDict(json.load(tf.io.gfile.GFile(ckpt_cfg_path, 'r')))
    if backbone == 'gemma_supervised':
      c.model_init = f"{backbone}/checkpoint.bv-{ckpt_cfg.total_steps:09d}"
      c.model = ckpt_cfg.model
    elif backbone == 'clip+llm':
      c.model_init['img'] = f"{backbone}/checkpoint.bv-{ckpt_cfg.total_steps:09d}:img"
      c.model.img = ckpt_cfg.model.image
      c.model.img['pool_type'] = 'none'

  # model_load
  dont_load = []
  if c.model.llm['head'] == 'map': dont_load += ['MAPHead.*']
  if c.model.llm['head'] == 'ffn': dont_load += ['FFNAdapter.*'] 
  if c.model.llm['projection']: dont_load += ['head/.*']
  c.model_load = {'img_load_kw': {}, 'llm_load_kw': {'dont_load': dont_load}}


  # Training Section

  # schedule


