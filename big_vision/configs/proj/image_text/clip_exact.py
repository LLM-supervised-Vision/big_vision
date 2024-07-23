import big_vision.configs.common as bvcc
from big_vision.configs.proj.image_text import common
from ml_collections import ConfigDict

def get_config(arg=None):
    arg = bvcc.parse_arg(
        arg, res=224, token_len=77, unified=False, memory_efficient=False, debug=True
    )
    config = ConfigDict()

    # Input section
    config.input = ConfigDict()
    config.input.batch_size = 32_768
    config.input.shuffle_buffer_size = 50_000
    config.input.data = dict(name='laion400m/images', split='train', data_dir='gs://us-central2-storage/tensorflow_datasets')

    if arg.unified: arg.token_len = 64
    tokenizer = lambda inkey, outkey: (
      f'tokenize(max_len={arg.token_len}, model="c4_en", clip_bpe={not arg.unified}, '
      f'eos="sticky", pad_value=1, inkey="{inkey}", outkey="{outkey}")'
    )
    config.input.pp = (
        f'decode|resize({arg.res})|flip_lr|value_range(-1,1)|'
        f'{tokenizer("caption", "labels")}|keep("image", "labels")'
    )

    # Model section
    config.model_name = 'proj.image_text.two_towers'
    config.model_load = {}
    config.model_init = None
    config.model = ConfigDict()
    config.model.image_model = 'vit'
    config.model.text_model = 'proj.image_text.text_transformer'
    config.model.image = dict(
        variant = 'B/16',
        posemb = 'learn',
        rep_size = False,
        dropout = 0.0,
        pool_type = 'gap',
        head_zeroinit = False,
        mask = None, # fully visible mask
        normalize_qk = False,
        scan = False,
        remat_policy = 'nothing_saveable',
        dtype_mm = 'float32',
        proj_bias = False,
    )
    config.model.text = dict(
        variant = 'M',
        dropout = 0.0,
        vocab_size = 49_408,
        pool_type = 'argmax',
        scan = False,
        remat_policy = 'nothing_saveable',
        dtype_mm = 'float32',
        normalize_qk = False,
        autoregressive = True,
        proj_bias = False,
        head_zeroinit = False,
    )
    config.model.out_dim = (768, 768)
    config.model.temperature_init = 1/0.07
    config.model.max_temperature = True
    config.model.bias_init = None

    # Training section
    config.total_steps = 91_553
    config.init_types = ['float32', 'int32']
    config.init_shapes = [(1, arg.res, arg.res, 3), (1, arg.token_len,)]

    config.loss_fn = 'softmax'
    config.lr = 5e-4
    config.wd = 2e-1
    config.optax_name = 'scale_by_adam'
    config.optax = dict(b1=0.9, b2=0.98, mu_dtype='bfloat16')
    config.schedule = [('.*', dict(decay_type='cosine', warmup_steps=2_000))]
    config.grad_clip_norm = 1.0

    config.ckpt_steps = 1000
    config.log_training_steps = 50
    config.wandb = True

    # Evaluation section
    config.evals = {}
    config.evals.retrieval_coco = common.get_coco(
        pp_img=f'resize({arg.res})|value_range(-1, 1)',
        pp_txt=tokenizer('texts','labels'),
        log_steps=1000,
    )
    config.evals.zeroshot_imagenet = common.get_disclf(
      sz=224, pp_txt=tokenizer('texts','labels'), 
      dataset_names=('imagenet2012','imagenet_v2','imagenet2012_real'),
      log_steps=1000,
    )

    # Memory efficient
    if arg.memory_efficient:
        config.model.image.scan = True
        config.model.text.scan = True
        config.model.image.dtype_mm = 'bfloat16'
        config.model.text.dtype_mm = 'bfloat16'
        config.mesh = [("data",-1)]
        config.sharding_strategy = [('.*', f'fsdp(axis="data", min_size_to_shard_mb=2)')]

    if arg.debug:
        config.input.data = dict(name='coco_captions', split='train', data_dir='gs://us-central2-storage/tensorflow_datasets')
        config.input.pp = (f'decode|resize({arg.res})|value_range(-1, 1)|'
                'coco_captions("captions")|choice(inkey="captions", outkey="text")|'
                f'{tokenizer("text", "labels")}|keep("image", "labels")')
        config.wandb = False

    if arg.unified:
        config.input.batch_size = 16_384
        config.input.shuffle_buffer_size = 250_000

        config.model.text.variant = 'B'
        config.model.text.vocab_size = 32_000
        config.model.text.pool_type = 'last'
        config.model.text.dtype_mm = 'bfloat16'
        config.model.image.dtype_mm = 'bfloat16'
        config.model.max_temperature = False

        config.total_steps = 183_105
        config.lr = 1e-3
        config.wd = 1e-4
        config.optax.b1 = 0.9
        config.optax.b2 = 0.95
        config.schedule = [('.*', dict(decay_type='cosine', warmup_steps=10_000))]

    return config