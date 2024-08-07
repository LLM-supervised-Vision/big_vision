import big_vision.configs.common as bvcc
from big_vision.configs.proj.image_text import common
from ml_collections import ConfigDict

def get_config(arg=None):
    arg = bvcc.parse_arg(
        arg, res=224, token_len=77, 
        loss_fn='softmax', unified=False, scale='small',
        lit=False, memory_efficient=False, debug=True
    )
    # common variables
    # TODO: Add more common variables here
    config = ConfigDict()

    # Input section
    config.input = ConfigDict()
    config.input.batch_size = 32_768
    config.input.shuffle_buffer_size = 50_000
    config.input.data = dict(name='laion400m/images', split='train', data_dir='gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets')

    if arg.unified: arg.token_len = 64
    tokenizer = lambda inkey, outkey: (
      f'tokenize(max_len={arg.token_len}, model="c4_en", clip_bpe={not arg.unified}, '
      f'eos="sticky", pad_value=1, inkey="{inkey}", outkey="{outkey}")'
    )
    config.input.pp = (
        f'decode|resize({arg.res})|flip_lr|value_range(-1,1)|'
        f'{tokenizer("caption", "labels")}|keep("image", "labels")'
    )
    config.input.pp_late = (f'{tokenizer("labels", "labels")}')

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
        pool_type = 'max',
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

    config.loss_fn = arg.loss_fn
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
        config.sharding_strategy = [('.*', f'fsdp(axis="data", min_size_to_shard_mb=1)')]

    if arg.debug:
        config.input.data = dict(name='coco_captions', split='train', data_dir='gs://us-central2-storage/tensorflow_datasets')
        config.input.pp = (f'decode|resize({arg.res})|value_range(-1, 1)|'
                'coco_captions("captions")|choice(inkey="captions", outkey="text")|'
                f'{tokenizer("text", "labels")}|keep("image", "labels")')

        config.input.batch_size = 16
        config.model.image.variant = 'mu/16'
        config.model.text.variant = 'mu'
        config.wandb = False

    if arg.unified or arg.loss_fn == 'sigmoid':
        config.input.batch_size = 16_384
        config.input.pp_late = ('')
        config.input.shuffle_buffer_size = 250_000

        config.model.text.variant = 'B'
        config.model.text.vocab_size = 32_000
        config.model.text.pool_type = 'last'
        config.model.text.autoregressive = False
        config.model.text.dtype_mm = 'bfloat16'
        config.model.image.dtype_mm = 'bfloat16'
        config.model.max_temperature = False
        if arg.loss_fn == 'sigmoid':
            config.model.out_dim = (None, 768)
            config.model.temperature_init = 10.0
            config.model.max_temperature = False
            config.model.bias_init = -10.0
            config.model.image.pool_type = 'map'

        config.total_steps = 183_105
        config.lr = 1e-3
        config.wd = 1e-4
        config.optax.b1 = 0.9
        config.optax.b2 = 0.95
        warmup_steps = max(int(0.03 * config.total_steps), 100)
        config.schedule = [('.*', dict(decay_type='cosine', warmup_steps=warmup_steps))]

    if arg.scale == 'large':
        config.input.batch_size = 32768
        config.model.image.variant = 'L/14'
        config.model.text.variant = 'L'
        config.total_steps = 390625

    if arg.lit:
        backbone = 'clip'
        backbone_dict = {
            'clip': 'gs://us-central2-storage/tensorflow_datasets/vit-b-16_3b_pretraining/clip_bs16384_warm10k_lr1e-3_wd1e-4_bf16_qknorm-F_b2-0.95_12lyr_07-23_1510/checkpoint.bv-000183105:img',
            'clip_map': 'gs://us-central2-storage/tensorflow_datasets/vit-b-16_3b_pretraining/clip_autoregressive_bs16384_warm0.03_lr1e-3_wd1e-4_bf16_qknorm-F_b2-0.95_12lyr_06-24_2019',
            'siglip': 'gs://us-central2-storage/tensorflow_datasets/vit-b-16_3b_pretraining/siglip_parallel_bs16384_warm0.03_lr1e-3_wd1e-4_bf16_qknorm-F_b2-0.95_12lyr_06-24_2019',
            'siglip_v4-32': 'gs://us-central2-storage/tensorflow_datasets/vit-b-16_3b_pretraining/siglip_replication_pod_04-11_2247',
            'cappa': 'gs://us-central2-storage/tensorflow_datasets/vit-b-16_3b_pretraining/cappa_bs16384_s3B_warm0.03_lr1e-3_wd1e-4_bf16_qknorm-F_b2-0.95_6lyr_06-27_2108/checkpoint.bv-000183106:encoder',
            'cappa_9k': 'gs://us-central2-storage/tensorflow_datasets/cappa_bs16384_s9B_warm0.02_lr1e-3_wd1e-4_bf16_qknorm-F_b2-0.95_6lyr_06-27_2108/checkpoint.bv-000549317:encoder'
            # 'cappa_decoder-qknorm-T_warm0.02': 'gs://us-central2-storage/tensorflow_datasets/cappa_bs16384_warm0.02_lr1e-3_wd1e-4_bf16_b2-0.95_6lyr_06-15_2102',
        }
        img_init = backbone_dict[backbone]
        config.model_init = {'image': img_init, 'text': None}

        dont_load = [
            'head/kernel', 'head/bias',
            'MAPHead_0/.*',
        ]
        config.model_load['img_load_kw'] = {'dont_load': dont_load}
        
        config.model.image.pool_type = 'map'
        config.model.image.dtype_mm = 'bfloat16'
        config.model.text = dict(
            variant = 'B',
            dropout = 0.0,
            vocab_size = 32_000,
            pool_type = 'gap',
            scan = False,
            remat_policy = 'nothing_saveable',
            dtype_mm = 'bfloat16',
            normalize_qk = False,
            autoregressive = False,
            proj_bias = False,
            head_zeroinit = False,
        )
        config.model.out_dim = (None, 768)

        config.input.batch_size = 32_768
        config.total_steps = 27465 # 0.9*1e9/32_768
        config.optax_name = 'scale_by_lion'
        # config.optax = dict(b1=0.9,b2=0.99,mu_dtype='bfloat16')
        config.lr = 1e-4
        config.wd = 1e-7
        warmup_steps = 6_500
        config.schedule = [
            ('img/.*', None), 
            ('.*', dict(decay_type='cosine', warmup_steps=warmup_steps)),
        ]

        

    return config