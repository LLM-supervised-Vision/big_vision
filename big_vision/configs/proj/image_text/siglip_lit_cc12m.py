# pylint: disable=line-too-long
r"""SigLIP (https://arxiv.org/abs/2303.15343) LiT with CC12M.

Example training:

big_vision.trainers.proj.image_text.siglip \
    --config big_vision/configs/proj/image_text/lit_coco.py:batch_size=512 \
    --workdir gs://$GS_BUCKET_NAME/big_vision/`date '+%Y-%m-%d_%H%M'`
"""

import big_vision.configs.common as bvcc
from big_vision.configs.proj.image_text import common
from ml_collections import ConfigDict


def get_config(arg=None):
  """The base configuration."""
  arg = bvcc.parse_arg(
      arg, res=224, runlocal=False, token_len=16, txt='bert_base', img='B/16',
      init='', img_head=False, batch_size=20480, total_samples=3.0, loss_fn="softmax",dtype='bfloat16',fsdp=2)
  # img_name, img_init = common.inits[arg.img]
  img_name, img_init = "B/16","gs://us-central2-storage/tensorflow_datasets/siglip_replication_pod_04-11_2247/checkpoint.bv-000183105:img"
  txt_name, txt_init = common.inits[arg.txt]
  config = ConfigDict()

  config.input = {}
  config.input.data = dict(name='cc12m', data_dir = "gs://us-central2-storage/tensorflow_datasets", split='train')
  config.input.batch_size = arg.batch_size if not arg.runlocal else 32
  config.input.shuffle_buffer_size = 250_000  if not arg.runlocal else 50

  def calculate_total_step(batch_size, total_samples):
    # the unit for total_samples is billion (e.g. 1 billion example corresponds to 1.0)
    total = total_samples * 1e9
    steps = total // batch_size
    if total%batch_size: steps += 1
    return int(steps)
  
  config.total_steps = calculate_total_step(arg.batch_size,arg.total_samples) if not arg.runlocal else 1

  config.init_shapes = [(1, arg.res, arg.res, 3), (1, arg.token_len,)]
  config.init_types = ['float32', 'int32']

  if arg.init:
    vocab_path = arg.init.rsplit('.', 1)[0] + '.txt'
  else:
    vocab_path = f'{txt_init}/vocab.txt'
  tokenizer = lambda inkey: (
      f'bert_tokenize(inkey="{inkey}", max_len={arg.token_len}, '
      f'vocab_path="{vocab_path}")')
  config.input.pp = (
      f'decode|resize({arg.res})|flip_lr|value_range(-1,1)'
      f'|flatten|{tokenizer("caption")}|keep("image", "labels")'
  )
  config.pp_modules = ['ops_general', 'ops_image', 'ops_text',
                       'proj.flaxformer.bert_ops']

  config.log_training_steps = 50
  config.ckpt_steps = 1000

  # Model section
  config.model_name = 'proj.image_text.two_towers'
  config.model_load = {}
  if arg.init:
    config.model_init = arg.init
  else:
    config.model_init = {'image': img_init, 'text': None}
    # config.model_init = {'image': img_init, 'text': txt_init}
    # config.model_load['txt_load_kw'] = {'dont_load': ['head/kernel', 'head/bias']}
    if not arg.img_head:
      config.model_load['img_load_kw'] = {'dont_load': ['head/kernel', 'head/bias']}
      # config.model_load['img_load_kw'] = {'dont_load': ['head/kernel', 'head/bias', 'MAPHead_0/*']}
  config.model = ConfigDict()
  config.model.image_model = 'vit'
  config.model.text_model = 'proj.flaxformer.bert'
  config.model.image = ConfigDict({
      'variant': img_name,
      'pool_type': 'map',
      'head_zeroinit': False,
      'scan': True,
      'dtype_mm': arg.dtype,
  })
  config.model.text = ConfigDict({
      'config': txt_name,
      'head_zeroinit': False,
      'dtype_mm': arg.dtype,
  })
  config.loss_fn = arg.loss_fn # softmax, sigmoid
  config.model.temperature_init = 10.0 if config.loss_fn == "sigmoid" else 1/0.07
  dim = {'B': 768, 'L': 1024}[arg.img[0]]
  config.model.out_dim = (dim if arg.img_head else None, dim)  # (image_out_dim, text_out_dim)
  config.model.bias_init = -2.71 if config.loss_fn == "sigmoid" else None # TO_DETERMINE: -10.0 or -2.71

  if txt_name == 'base':
    config.optax_name = 'scale_by_adam'
    config.optax = dict(b1=0.9, b2=0.999,mu_dtype=arg.dtype)
  else:
    config.optax_name = 'big_vision.scale_by_adafactor'

  config.mesh = [("data",-1)]
  config.sharding_strategy = [('.*', f'fsdp(axis="data", min_size_to_shard_mb={arg.fsdp})')]

  config.lr = 1e-3
  config.wd = 1e-4
  warmup_steps = max(int(0.03 * config.total_steps), 100)
  config.schedule = [
      ('img/.*', None),  # Freezes image tower.
      # ('img/MAPHead_0/.*', dict(decay_type='consine', warmup_steps=warmup_steps)),
      ('.*', dict(decay_type='cosine', warmup_steps=warmup_steps)),
  ]

  config.grad_clip_norm = 1.0

  config.evals = {}
  config.evals.retrieval_coco = common.get_coco(
      pp_img=f'resize({arg.res})|value_range(-1, 1)',
      pp_txt=tokenizer('texts'),
      log_steps=1000,
  )
  # config.evals.zeroshot_imagenet = common.get_disclf(
  #   sz=224, pp_txt=tokenizer('texts'), 
  #   dataset_names=('imagenet2012'),
  #   log_steps=1000,
  # )


  return config