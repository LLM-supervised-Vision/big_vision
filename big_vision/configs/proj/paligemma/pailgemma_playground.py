import big_vision.configs.common as bvcc
from ml_collections import ConfigDict

def training_data(res, *, final_split, prefix, text_len=32):
  """Creates training data config.

  You can add more arguments beside `res`, but give them good defaults.
  
  Args:
    res: The requested image resolution (eg 224).
    final_split: Train on all train+val data.
    text_len: sequence length.

  Returns:
    The ConfigDict for the input section.
  """
  c = bvcc.parse_arg('')  # Just make a configdict without extra import.
  c.data = dict(
      name='laion400m/images',
      split='train' if final_split else 'train',
      data_dir='gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets'
  )
  # c.pp = '|'.join([
      # f'decode|resize({res})|value_range(-1, 1)',
      
  return c



def get_config(arg=None):
  c = bvcc.parse_arg(
      arg, res=224,
      freeze_vit=False, freeze_llm=True,
  )
  c.name = 'what the hell is this???'

  # Input section
  c.input = ConfigDict()
  c.input.data = dict(
      name='laion400m/images',
      split='train',
      data_dir='gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets'
  )
  tokenizer = lambda inkey, outkey: (
      f'tokenize(max_len={64}, model="c4_en", clip_bpe=False, '
      f'eos="sticky", pad_value=1, inkey="{inkey}", outkey="{outkey}")'
  )
  c.input.pp = (
      f'decode|resize({c.res})|flip_lr|value_range(-1,1)|'
      f'{tokenizer("caption", "labels")}|keep("image", "labels")'
  )

  c.total_epochs = 1
  c.input.batch_size = 256
  c.optax_name = 'scale_by_adam'
  c.optax = dict(b2=0.999)
  c.lr = 1e-5
  c.wd = 0.0
  c.grad_clip_norm = 1.0
  c.label_smoothing = 0.0

  # Learning-rate schedule. Probably is fine like this.
  sched = dict(decay_type='cosine', warmup_percent=0.05)
  c.schedule = [
      ('img/.*', None if c.freeze_vit else sched),
      ('llm/.*', None if c.freeze_llm else sched),
  ]


  # Model section.
  c.model_name = 'proj.paligemma.paligemma'
  c.model = {}
  # c.model.img = dict(variant='So400m/14', pool_type='none', scan=True)
  c.model.img = dict(variant='So400m/14', pool_type='none', scan=True)
  c.model.llm = dict(vocab_size=256_000 + 1024 + 128, dropout=0.0)
  c.model_init = ''
  # c.model_init = f'pt_{c.res}'

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
  return c
