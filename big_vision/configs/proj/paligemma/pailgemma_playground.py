import big_vision.configs.common as bvcc
from ml_collections import ConfigDict
from big_vision.configs.proj.paligemma.transfers.common import combine_and_keep_train, combine_and_keep_eval, TOKENIZER

def training_data(res, *, prefix, text_len=32):
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


def get_config(arg=None):
  c = bvcc.parse_arg(
      arg, res=224,
      mode='generative', freeze_vit=False, freeze_llm=True,
      batch_size=8192, total_samples=3.0, debug=False,
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
  c.wd = 0.1
  c.grad_clip_norm = 1.0
  c.label_smoothing = 0.0

  # Learning-rate schedule. Probably is fine like this.
  sched = dict(decay_type='cosine', warmup_percent=0.05)
  c.schedule = [
      ('img/.*', None if c.freeze_vit else sched),
      ('llm/.*', None if c.freeze_llm else sched),
      ('t', sched),
  ]

  # Model section.
  c.model_name = 'proj.paligemma.paligemma'
  c.model = {}
  c.model.img = dict(variant='So400m/14', pool_type='none', head_zeroinit=False, scan=True)
  c.model.llm = dict(vocab_size=256_000 + 1024 + 128, dropout=0.0, scan=True)
  # c.model_init = f'pt_{c.res}'
  c.model_init = {'img': None, 'llm': '/home/austinwang/gemma2b.npz'}

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

  if c.debug:
    c.input.shuffle_buffer_size = None
    c.input.batch_size = 32
    c.total_steps = 10
    c.log_training_steps = 1

  return c
