# pylint: disable=line-too-long
r"""Trains a CoCa model (https://arxiv.org/pdf/2205.01917) on laion400m/images.

bash /home/austinwang/austin_big_vision/scripts/coca.sh
"""


from big_vision.configs.proj.image_text import common
from big_vision.configs import common_fewshot
import big_vision.configs.common as bvcc
import ml_collections


def get_config(arg=None):
  """Returns the base config."""
  config = bvcc.parse_arg(arg,
                          runlocal=False,
                          contrastive_weight=1.0,
                          captioning_weight=1.0,
                          total_steps=366_500,
                          batch_size=8*1024,
                          total_samples=3.0,
                          warmup_steps=10_000,
                          warmup_ratio=0.15,
                          lr=1e-4,
                          wd=1e-4,
                          dtype='float32',
                          dec_lyr=12,
                          masked_pred_prob=0.0,
                          res=224,
                          scan=True,
                          eval_only=False,
                          debug=False,
                          )

  config.evals = {}
  config.input = {}
  config.input.batch_size = config.batch_size if not config.runlocal else 8
  config.total_steps = int(config.total_samples*1e9 / config.batch_size) if not config.runlocal else 1
  shuffle_buffer_size = 50_000 if not config.runlocal else 50

  res = 224
  patch_size = 16
  max_text_tokens = 64
  add_bos = False

  pp_image = (f'resize({res})|value_range(-1,1)')

  def tokenizer(inkey, outkey):
    return (f'tokenize(max_len={max_text_tokens}, model="c4_en", '
            f'eos="sticky", add_bos={add_bos}, inkey="{inkey}", outkey="{outkey}")')

  pp_laion = (f'decode|{pp_image}|'
              'choice(inkey="caption", outkey="text")|'
              f'{tokenizer("text", "labels")}|keep("image", "labels")')
  config.input.pp = pp_laion
  config.input.data = dict(name='laion400m/images', split='train', data_dir='gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets')  # num_examples=379,600,897
  config.input.shuffle_buffer_size = shuffle_buffer_size

  pp_coco = (f'decode|{pp_image}|'
             'coco_captions("captions")|choice(inkey="captions", outkey="text")|'
             f'{tokenizer("text", "labels")}|keep("image", "labels")')

  config.contrastive_weight = config.contrastive_weight
  config.captioning_weight = config.captioning_weight
  config.log_steps = 1000

  if config.get('contrastive_weight', 0.0) != 0.0:
    config.evals.retrieval_coco = common.get_coco(
        pp_img=f'resize({config.res})|value_range(-1, 1)',
        pp_txt=tokenizer('texts','labels'),
        log_steps=config.log_steps,
    )
    config.evals.zeroshot_imagenet = common.get_disclf(
      sz=224, pp_txt=tokenizer('texts','labels'), 
      dataset_names=('imagenet2012','imagenet_v2','imagenet2012_real'),
      log_steps=config.log_steps,
    )

  if config.get('captioning_weight', 0.0) != 0.0:
    config.evals.val_coco = {
        'type': 'proj.cappa.perplexity',
        'pred': 'perplexity',
        'log_steps': config.log_steps,
        'data': dict(name='coco_captions', split='val'),  # num_examples=5_000
        'pp_fn': pp_coco,
    }
    # # Few-shot  metrics
    # config.evals.fewshot = common_fewshot.get_fewshot_lsr(
    #     target_resolution=res, resize_resolution=int(256 / 224 * res))
    # config.evals.fewshot.type = 'fewshot_lsr'
    # config.evals.fewshot.log_steps = config.log_steps if not config.runlocal else 5
    # config.evals.fewshot.representation_layer = 'pre_logits'
    # config.evals.fewshot.pred = 'enc_rep'
    # config.evals.fewshot.pp_eval = config.evals.fewshot.pp_train

    # NOTE: Scoring of the entire imagenet validation set is rather slow:
    # ~100 secs / 1k classes / host.
    config.evals['imagenet/scoring'] = dict(
        type='proj.cappa.scoring_classifier',
        pred='score',
        log_percent=0.05,
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
  config.model_name = 'proj.coca.coca'
  config.model = ml_collections.ConfigDict()
  config.model.num_layers = 12
  config.model.num_heads = 12
  config.model.mlp_dim = 3072
  config.model.emb_dim = 768
  config.model.vocab_size = 32_000
  config.model.patches = (patch_size, patch_size)
  config.model.seq_len = max_text_tokens
  config.model.posemb_type = 'learn'
  config.model.pool_type = f'map:{(config.res//patch_size)**2}'

  # Decoder
  config.model.decoder_num_layers = config.dec_lyr
  # 0 values here mean to use the same value as for the encoder
  config.model.decoder_num_heads = 0
  config.model.decoder_mlp_dim = 0
  config.model.decoder_emb_dim = 0
  config.model.dec_dropout_rate = 0.0
  config.model.masked_pred_prob = config.masked_pred_prob
  config.model.masking_ratio = 1.0
  config.model.decoder_bias = True

  config.model.scan = config.scan
  config.model.dtype_mm = config.dtype # 'bfloat16'
  config.model.temperature_init = 1.0/0.07

  # config.mesh = [("fsdp",-1)]
  # config.sharding_strategy = [('.*', f'fsdp(axis="fsdp", min_size_to_shard_mb=2)')]

  # config.optax_name = 'big_vision.scale_by_adafactor'
  # config.optax = dict(beta2_cap=0.95)
  config.optax_name = 'scale_by_adam'
  config.optax = dict(b1=0.9, b2=0.95, mu_dtype='bfloat16')
  config.grad_clip_norm = 1.0
  config.label_smoothing = 0.0

  config.warmup_steps = max(int(config.warmup_ratio * config.total_steps), 100)
  schedule = [('.*',dict(decay_type='cosine',
                  warmup_steps=config.warmup_steps
                  if not config.runlocal else 5))]

  # Standard schedule
  config.lr = config.lr # 5e-5
  config.wd = config.wd # 0.01
  config.schedule = schedule

  config.seed = 0
  config.wandb = not config.debug
  if config.debug:
    # replace the data and pp with coco_captions for faster debugging
    config.input.data = dict(name='coco_captions', split='train', data_dir='gs://us-central2-storage/tensorflow_datasets')
    pp_coco = (f'decode|{pp_image}|'
              'coco_captions("captions")|choice(inkey="captions", outkey="text")|'
              f'{tokenizer("text", "labels")}|keep("image", "labels")')
    config.input.pp = pp_coco
    config.input.shuffle_buffer_size = None
    config.input.batch_size = 32
    config.total_steps = 10
    config.log_training_steps = 1
    config.schedule = [('.*',dict(decay_type='cosine', warmup_steps=3))]
    config.evals = {}

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

    # config.mesh = [('data', -1)]
    # config.sharding_strategy = [('params/.*', 'fsdp(axis="data")')]
    # config.sharding_rules = [('act_batch', ('data',))]

  return config