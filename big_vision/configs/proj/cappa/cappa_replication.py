# pylint: disable=line-too-long
r"""Trains a CapPa model (https://arxiv.org/abs/2306.07915) on laion400m/images.

bash /home/austinwang/austin_big_vision/scripts/cappa.sh
"""


from big_vision.configs import common_fewshot
import big_vision.configs.common as bvcc
import ml_collections


def get_config(arg=None):
  """Returns the base config."""
  config = bvcc.parse_arg(arg,
                          runlocal=False,
                          total_steps=366_500,
                          batch_size=8*1024,
                          warmup_steps=10_000,
                          total_samples=9.0,
                          eval_only=False
                          )

  config.evals = {}
  config.input = {}
  config.input.batch_size = config.batch_size if not config.runlocal else 8
  shuffle_buffer_size = 50_000 if not config.runlocal else 50

  # num_tpu_chips/samples_seen/batch_size->ETA,ETA(ckpting): 4/3B/512->17d8h; 4/3B/1024->14d12h,17d; 4/3B/2048->OOM; 4/3B/4096->OOM,18d6h; 4/3B/32_768->OOM
  def calculate_total_step(batch_size, total_samples_seen):
    # the unit for total_samples_seen is billion (e.g. 1 billion example corresponds to 1.0)
    total = total_samples_seen * 1e9
    steps = total // batch_size
    if total%batch_size: steps += 1
    return int(steps)
  
  config.total_steps = calculate_total_step(config.batch_size,config.total_samples) if not config.runlocal else 1

  res = 224
  patch_size = 16
  max_text_tokens = 64

  pp_image = (f'resize({res})|value_range(-1,1)')

  def tokenizer(inkey, outkey):
    return (f'tokenize(max_len={max_text_tokens}, model="c4_en", '
            f'eos="sticky", inkey="{inkey}", outkey="{outkey}")')

  pp_laion = (f'decode|{pp_image}|'
              'choice(inkey="caption", outkey="text")|'
              f'{tokenizer("text", "labels")}|keep("image", "labels")')
  config.input.pp = pp_laion
  config.input.data = dict(name='laion400m/images', split='train')  # num_examples=379,600,897
  config.input.shuffle_buffer_size = shuffle_buffer_size

  pp_coco = (f'decode|{pp_image}|'
             'coco_captions("captions")|choice(inkey="captions", outkey="text")|'
             f'{tokenizer("text", "labels")}|keep("image", "labels")')
  config.evals.val_coco = {
      'type': 'proj.cappa.perplexity',
      'pred': 'perplexity',
      'log_steps': 1000,
      'data': dict(name='coco_captions', split='val'),  # num_examples=5_000
      'pp_fn': pp_coco,
  }

  # # Few-shot  metrics
  # config.evals.fewshot = common_fewshot.get_fewshot_lsr(
  #     target_resolution=res, resize_resolution=int(256 / 224 * res))
  # config.evals.fewshot.type = 'fewshot_lsr'
  # config.evals.fewshot.log_steps = 5_000 if not config.runlocal else 5
  # config.evals.fewshot.representation_layer = 'pre_logits'
  # config.evals.fewshot.pred = 'enc_rep'
  # config.evals.fewshot.pp_eval = config.evals.fewshot.pp_train

  # NOTE: Scoring of the entire imagenet validation set is rather slow:
  # ~100 secs / 1k classes / host.
  config.evals['imagenet/scoring'] = dict(
      type='proj.cappa.scoring_classifier',
      pred='score',
      log_percent=0.1,
      data=dict(name='imagenet2012', split='validation'),
      pp_fn=f'decode|{pp_image}|keep("image", "label")',
      pp_txt=tokenizer('label', 'labels'),
  )

  for e in config.evals.values():
    e.skip_first = True

  config.log_training_steps = 50
  config.ckpt_steps = 1000
  config.keep_ckpt_steps = None  # 10_000

  # Model section
  config.model_name = 'proj.cappa.cappa'
  config.model = ml_collections.ConfigDict()
  config.model.num_layers = 12
  config.model.num_heads = 12
  config.model.mlp_dim = 3072
  config.model.emb_dim = 768
  config.model.vocab_size = 32_000
  config.model.patches = (patch_size, patch_size)
  config.model.seq_len = max_text_tokens
  config.model.posemb_type = 'learn'
  config.model.scan = True
  config.model.dtype_mm = 'bfloat16'

  # Decoder
  config.model.decoder_num_layers = 6
  # 0 values here mean to use the same value as for the encoder
  config.model.decoder_num_heads = 0
  config.model.decoder_mlp_dim = 0
  config.model.decoder_emb_dim = 0
  config.model.dec_dropout_rate = 0.0
  config.model.masked_pred_prob = 0.75
  config.model.masking_ratio = 1.0
  config.model.decoder_bias = False

  config.optax_name = 'big_vision.scale_by_adafactor'
  config.optax = dict(beta2_cap=0.95)
  config.grad_clip_norm = 1.0
  config.label_smoothing = 0.0

  warmup_steps = max(int(0.02 * config.total_steps), 100)
  schedule = dict(decay_type='cosine',
                  warmup_steps=warmup_steps
                  if not config.runlocal else 5)

  # Standard schedule
  config.lr = 0.001
  config.wd = 0.0001
  config.schedule = schedule
  config.wandb = True

  config.seed = 0

  if config.eval_only:
    config.total_steps = 0
    config.input = {}
    config.input.batch_size = config.batch_size if not config.runlocal else 8
    config.input.data = dict(name='coco_captions', split='train', data_dir='gs://us-central2-storage/tensorflow_datasets')
    pp_coco = (f'decode|{pp_image}|'
              'coco_captions("captions")|choice(inkey="captions", outkey="text")|'
              f'{tokenizer("text", "labels")}|keep("image", "labels")')
    config.input.pp = pp_coco
    config.optax_name = 'identity'
    config.optax = {}
    config.lr = 0.0

  return config