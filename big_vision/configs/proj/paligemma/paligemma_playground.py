from ml_collections import ConfigDict
import big_vision.configs.common as bvcc
from big_vision.configs.proj.image_text import common
from big_vision.configs.proj.paligemma.transfers.common import combine_and_keep_train, combine_and_keep_eval, TOKENIZER

def training_data(res, *, prefix, text_len=64):
  """Creates training data config.

  You can add more arguments beside `res`, but give them good defaults.
  
  Args:
    res: The requested image resolution (eg 224).
    text_len: sequence length.

  Returns:
    The ConfigDict for the input section.
  """
  c = bvcc.parse_arg('')  # Just make a configdict without extra import.
  c.data = dict(
      name='laion400m/images',
      split='train',
      data_dir='gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets'
  )
  c.pp = '|'.join([
      f'decode|resize({res})|value_range(-1,1)',
      f'strfmt("{prefix}", outkey="prefix")',
      'copy(inkey="caption", outkey="suffix")',
      combine_and_keep_train(text_len),
  ])
  return c

def add_eval(c, res, *, text_len=64, prefix, **kw):
  c.evals.retrieval_coco = common.get_coco(
    pred='contrastive_logits',
    pp_img=f'resize({res})|value_range(-1, 1)',
    pp_txt='|'.join([
      f'strfmt("{prefix}", outkey="prefix")',
      'copy(inkey="texts", outkey="suffix")',
      combine_and_keep_eval(text_len),
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
      combine_and_keep_eval(text_len),
      f'copy(inkey="text", outkey="labels")',
    ]),
    dataset_names=('imagenet2012','imagenet_v2','imagenet2012_real'),
    log_steps=1000,
  )
  c.evals.zeroshot_imagenet.update(kw)



def get_config(arg=None):
  c = bvcc.parse_arg(
      arg, res=224,
      mode='generative', freeze_vit=False, freeze_llm=True, half_llm=False,
      batch_size=8192, total_samples=3.0, debug=False, dtype='float32'
  )
  c.name = 'what the hell is this???'

  # Input section
  c.input = training_data(c.res, prefix='', text_len=64)

  # c.total_epochs = 1
  c.input.batch_size = c.batch_size
  c.total_steps = int(c.total_samples*1e9 / c.input.batch_size)
  c.optax_name = 'scale_by_adam'
  c.optax = dict(b1=0.9,b2=0.95)
  c.lr = 1e-3
  c.wd = 1e-4
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
      ('llm/.*', 0.1),
      ('t', 1.0),
    ]

  # Model section.
  c.model_name = 'proj.paligemma.paligemma'
  c.model = {}
  c.model.img = dict(variant='B/16', pool_type='none', head_zeroinit=False, scan=True, dtype_mm=c.dtype)
  c.model.llm = dict(vocab_size=256_000 + 1024 + 128, dropout=0.0, scan=True, dtype=c.dtype)
  # c.model_init = f'pt_{c.res}'
  c.model_init = {'img': None, 'llm': '/home/austinwang/gemma2b.npz'}
  if c.half_llm:
    c.model.llm['variant'] = 'gemma_2b_half'
    c.model_init['llm'] = '/home/austinwang/gemma2b_half.npz'
    dont_load = ['final_norm/scale']
    c.model_load = {'llm_load_kw': {'dont_load': dont_load}}


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
  if c.mode == 'contrastive':
    c.evals = {}
    add_eval(c, c.res, prefix='', batch_size=1024)

  if c.debug:
    c.input.shuffle_buffer_size = None
    c.input.batch_size = 32
    c.total_steps = 10
    c.log_training_steps = 1
    c.evals = {}

  return c
