import os
import logging
import json
import tensorflow as tf
import ml_collections
from ml_collections import ConfigDict

import big_vision.configs.common as bvcc
from big_vision.configs.proj.image_text import common
from big_vision.configs.proj.paligemma.transfers.common import combine_and_keep_train, combine_and_keep_eval, TOKENIZER

def training_data(res, *, prefix, text_len=64,dataset_name='laion400m/images',datacomp_inkey='re_caption'):
  """Creates training data config.

  You can add more arguments beside `res`, but give them good defaults.
  
  Args:
    res: The requested image resolution (eg 224).
    text_len: sequence length.

  Returns:
    The ConfigDict for the input section.
  """
  match dataset_name.split("/")[0]:
    case 'laion400m':
      inkey = 'caption'
    case 'datacomp_recap':
      inkey = datacomp_inkey
    case _:
      raise ValueError(f"Unknown dataset_name: {dataset_name}")
  c = bvcc.parse_arg('')  # Just make a configdict without extra import.
  c.data = dict(
      name=dataset_name,
      split='train',
      data_dir='gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets'
  )
  c.pp = '|'.join([
      f'decode|resize({res})|value_range(-1,1)',
      f'strfmt("{prefix}", outkey="prefix")',
      f'copy(inkey="{inkey}", outkey="suffix")',
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



def get_config(arg=None):
  c = bvcc.parse_arg(
      arg, res=224,
      mode='generative', loss_fn='softmax', dataset_name='laion400m/images', datacomp_inkey='re_caption',datacomp_backbone='gemma_supervised',drop_path_rate=0.0, lr=1e-3, wd=1e-4, epoch=5,
      freeze_vit=False, img_variant='B/16', img_beit_init=False, img_qknorm=False,
      freeze_llm=True, llm_variant='gemma_2b',llm_ckpt="full", llm_pool='none', llm_lr_mult=0.1, llm_dropout=0.0, llm_clean_vocab=False, llm_projection=False,
      batch_size=8192, total_samples=3.0, dtype='float32',
      debug=False, 
  )
  c.name = 'what the hell is this???'

  # Input section
  c.input = training_data(c.res, prefix='', text_len=64, dataset_name=c.dataset_name, datacomp_inkey=c.datacomp_inkey) # laion400m/images, datacomp_recap/10M:1.0.0

  # c.total_epochs = 1
  c.input.batch_size = c.batch_size
  c.total_steps = int(c.total_samples*1e9 / c.input.batch_size)
  c.optax_name = 'scale_by_adam'
  c.optax = dict(b1=0.9,b2=0.95)
  c.lr = c.lr
  c.wd = c.wd
  c.grad_clip_norm = 1.0
  c.label_smoothing = 0.0

  # Learning-rate schedule. Probably is fine like this.
  sched = dict(decay_type='cosine', warmup_percent=0.03)
  c.schedule = [
      ('img/.*', None if c.freeze_vit else sched),
      ('llm/.*', None if c.freeze_llm else sched),
      ('t', sched),
  ]
  if not c.freeze_vit and not c.freeze_llm:
    c.lr_mults = [
      ('img/.*', 1.0),
      ('llm/.*', c.llm_lr_mult),
      ('t', 1.0),
    ]

  # Model section.
  c.model_name = 'proj.paligemma.paligemma'
  c.model = {}
  c.model.temperature_init = 1/0.07 if c.loss_fn == 'softmax' else 10.0
  c.model.bias_init = None if c.loss_fn == 'softmax' else -10.0
  c.model.img = dict(
    variant=c.img_variant, scan=True, dtype_mm=c.dtype,
    pool_type='none', head_zeroinit=False, beit_init=c.img_beit_init, 
    drop_path_rate=c.drop_path_rate, normalize_qk=c.img_qknorm,
  )
  c.model.llm = dict(
    variant=c.llm_variant,scan=True, dtype=c.dtype, 
    dropout=c.llm_dropout, lyrs_frozen=-1, pool=c.llm_pool, projection=c.llm_projection,
    drop_path_rate=c.drop_path_rate, remat_policy='nothing_saveable',
  )
  if c.llm_clean_vocab == False:
    c.model['llm']['vocab_size'] = 256_000 + 1024 + 128

  dont_load = []
  if c.model.llm['pool'] == 'map': dont_load += ['MAPHead.*']
  if c.model.llm['projection']: dont_load += ['head/.*']
  c.model_load = {'img_load_kw': {}, 'llm_load_kw': {'dont_load': dont_load}}

  llm_ckpt = '/home/austinwang/gemma2b.npz'
  if ':' in c.llm_ckpt: c.llm_ckpt, lyrs_frozen = c.llm_ckpt.split(':')
  match c.llm_ckpt:
    case 'full':
      pass
    case 'half':
      c.model.llm['variant'] = 'gemma_2b_half'
      llm_ckpt = '/home/austinwang/gemma2b_half.npz'
      c.model_load['llm_load_kw']['dont_load'] += ['final_norm/scale']
    case '6lyr':
      c.model.llm['variant'] = 'gemma_6lyr'
      llm_ckpt = '/home/austinwang/gemma2b_first_6.npz'
      c.model_load['llm_load_kw']['dont_load'] += ['final_norm/scale']
    case 'partial_frozen':
      c.model.llm['lyrs_frozen'] = int(lyrs_frozen)
      assert c.freeze_llm==False, "partial_frozen is for unfreezing"
      c.schedule = [
        ('img/.*', None if c.freeze_vit else sched),
        ('llm/layers/frozen/.*', None),
        ('.*', sched),
      ]
    case 'ln':
      # unfreeze the layer norm only for llm
      assert c.freeze_llm==False, "ln is for unfreezing"
      c.schedule = [
        ('img/.*', None if c.freeze_vit else sched),
        ('.*norm/.*', sched),
        ('.*', None),
      ]
    case 'scratch':
      llm_ckpt = None
      c.model_init = None
      c.model_load = {}
    case 'scratch-partial_frozen':
      llm_ckpt = None
      c.model_init = None
      c.model_load = {}
      c.model.llm['lyrs_frozen'] = int(lyrs_frozen)
      assert c.freeze_llm==False, "scratch-partial_frozen is for unfreezing"
      c.schedule = [
        ('img/.*', None if c.freeze_vit else sched),
        ('llm/layers/frozen/.*', None),
        ('.*', sched),
      ]
    case _:
      raise ValueError(f"Unknown llm_ckpt: {c.llm_ckpt}")
  
  if llm_ckpt is not None: 
    c.model_init = {'img': None, 'llm': llm_ckpt}
    # check whether llm_ckpt exists or not, if not then download it from gcs directory
    if not os.path.exists(llm_ckpt):
      gcs_dir = 'gs://us-central2-storage/tensorflow_datasets/'
      gcs_path = gcs_dir + llm_ckpt.split('/')[-1]
      logging.info(f"Downloading {gcs_path} to {llm_ckpt}")
      os.system(f'gsutil cp {gcs_path} {llm_ckpt}')

  # c.model_init = '/mnt/vlm-pd/ckpts/paligemma/paligemma-3b-pt-224.bf16.npz'
  # /mnt/vlm-pd/ckpts/paligemma/paligemma-3b-pt-224.bf16.npz
  # /mnt/vlm-pd/ckpts/paligemma/paligemma-3b-pt-224.f16.npz
  # /mnt/vlm-pd/ckpts/paligemma/paligemma-3b-pt-224.npz

  # FSDP strategy.
  c.mesh = [('data', -1)]
  c.sharding_strategy = [('.*', 'fsdp(axis="data")')]
  c.sharding_rules = [('act_batch', ('data',))]

  # These probably do not need any change/tuning
  c.input.shuffle_buffer_size = 50_000
  c.log_training_steps = 50
  c.ckpt_steps = 1_000
  c.pp_modules = ['ops_general', 'ops_image', 'ops_text', 'proj.paligemma.ops']
  c.seed = 0
  c.wandb = not c.debug

  # Evaluation section
  c.evals = {}
  add_eval(c, c.res, prefix='', batch_size=1024, mode=c.mode)

  if c.dataset_name.split("/")[0] == 'datacomp_recap':
    assert "M" in c.dataset_name, "datacomp_recap dataset_name should have M in it"
    epochs = c.epoch
    match c.dataset_name.split("/")[1].split(":")[0]:
      case '10M':
        num_samples = 8344225
      case '50M':
        num_samples = 41598460
      case _:
        raise ValueError(f"Unknown dataset_name: {c.dataset_name}")
    c.total_steps = int(num_samples * epochs / c.input.batch_size)
    c.schedule = [('.*', dict(decay_type='cosine', warmup_steps=int(0.03*c.total_steps)))]

    if c.datacomp_backbone == 'gemma_supervised':
      backbone = "gs://us-central2-storage/tensorflow_datasets/mllm_ckpts/paligemma/gemma2b-partial_frozen99-0.01-gap_b16-F_contrastive_bs16k_s3b_lr1e-3_wd1e-4_bf16_09-01_0446"
      ckpt_cfg_path = f'{backbone}/config.json'
      ckpt_cfg = ml_collections.ConfigDict(json.load(tf.io.gfile.GFile(ckpt_cfg_path, 'r')))
      c.model_init = f"{backbone}/checkpoint.bv-{ckpt_cfg.total_steps:09d}"
      c.model = ckpt_cfg.model
    elif c.datacomp_backbone == 'clip+llm':
      backbone = "gs://us-central2-storage/tensorflow_datasets/vit-b-16_3b_pretraining/clip_bs16384_warm0.03_lr1e-3_wd1e-4_bf16_qknorm-F_b2-0.95_12lyr_07-25_1415"
      ckpt_cfg_path = f'{backbone}/config.json'
      ckpt_cfg = ml_collections.ConfigDict(json.load(tf.io.gfile.GFile(ckpt_cfg_path, 'r')))
      c.model_init['img'] = f"{backbone}/checkpoint.bv-{ckpt_cfg.total_steps:09d}:img"
      c.model.img = ckpt_cfg.model.image
      c.model.img['pool_type'] = 'none'
      if c.model_load is None: 
        c.model_load = {'img_load_kw': {'dont_load': ['head/.*']}}
      else:
        c.model_load['img_load_kw'] = {'dont_load': ['head/.*']}


  if c.debug:
    c.input.shuffle_buffer_size = None
    c.input.batch_size = 32
    c.total_steps = 10
    c.schedule = [('.*', dict(decay_type='cosine', warmup_steps=3))]
    c.log_training_steps = 1

    eval_when_debugging = False
    if eval_when_debugging:
      for k in c.evals: c.evals[k]['batch_size'] = 32
    else:
      c.evals = {}

  return c
