"""Configuration for PaLiGeMMA training."""

import big_vision.configs.common as bvcc
import big_vision.configs.proj.paligemma.config_utils as config_utils

def get_config(arg=None):
    """Creates the main configuration."""
    config = bvcc.parse_arg(
        arg,
        # Default values
        res=224,
        mode='generative',
        loss_fn='softmax',
        dataset_name='laion400m/images',
        drop_path_rate=0.0,
        lr=1e-3,
        wd=1e-4,
        org_caption_ratio=1.0,
        datacomp_backbone='gemma_supervised',
        epoch=-1.0,
        total_steps=-1,
        train_mode='pretrain',  # or 'finetune'
        vit_backbone=None,
        
        # Vision model params
        freeze_vit=False,
        img_variant='B/16',
        img_beit_init=False,
        img_qknorm=False,
        
        # Language model params
        freeze_llm=True,
        llm_variant='gemma_2b',
        llm_ckpt="full", # partial_frozen:9
        llm_head='none',
        llm_lr_mult=0.1,
        llm_dropout=0.0,
        llm_clean_vocab=False,
        llm_projection=False,
        
        # Training params
        batch_size=8192,
        total_samples=3.0,
        dtype='float32',
        
        # Debug options
        debug=False,
        # (only used when debug=True)
        debug_eval_only=False,
        debug_eval_when_debugging=False,
        debug_tiny_model=False,
    )

    # Setup input pipeline
    config.llm_text_len = config_utils.get_text_length(config.dataset_name, config.org_caption_ratio)
    config.input = config_utils.create_training_data_config(
        config.res,
        prefix='',
        dataset_name=config.dataset_name,
        org_caption_ratio=config.org_caption_ratio
    )

    # Training parameters
    config.input.batch_size = config.batch_size
    
    # Calculate total steps
    config.total_steps = config_utils.calculate_total_steps(config)
    
    # Optimizer configuration
    config.optax_name = 'scale_by_adam'
    config.optax = {'b1': 0.9, 'b2': 0.95}
    config.lr = config.lr
    config.wd = config.wd
    config.grad_clip_norm = 1.0
    config.label_smoothing = 0.0

    # Learning rate schedule
    config.schedule_base = {'decay_type': 'cosine', 'warmup_percent': 0.03}
    config.schedule = [
        ('img/.*', None if config.freeze_vit else config.schedule_base),
        ('llm/.*', None if config.freeze_llm else config.schedule_base),
        ('t', config.schedule_base),
    ]

    # Setup model initialization and schedule based on training mode
    config_utils.setup_model_init_and_schedule(config)

    # Model configuration
    config.model_name = 'proj.paligemma.paligemma'
    config.model = {
        'temperature_init': 1/0.07 if config.loss_fn == 'softmax' else 10.0,
        'bias_init': None if config.loss_fn == 'softmax' else -10.0,
        'img': {
            'variant': config.img_variant,
            'scan': True,
            'dtype_mm': config.dtype,
            'pool_type': 'none',
            'head_zeroinit': False,
            'beit_init': config.img_beit_init,
            'drop_path_rate': config.drop_path_rate,
            'normalize_qk': config.img_qknorm,
        },
        'llm': {
            'variant': config.llm_variant,
            'scan': True,
            'dtype': config.dtype,
            'dropout': config.llm_dropout,
            'lyrs_frozen': -1,
            'head': config.llm_head,
            'projection': config.llm_projection,
            'drop_path_rate': config.drop_path_rate,
            'remat_policy': 'nothing_saveable',
        }
    }
    
    if not config.llm_clean_vocab:
        config.model['llm']['vocab_size'] = 256_000 + 1024 + 128

    # Setup model initialization and loading
    config_utils.setup_model_config(config)

    # FSDP strategy configuration
    config.mesh = [('data', -1)]
    config.sharding_strategy = [('.*', 'fsdp(axis="data")')]
    config.sharding_rules = [('act_batch', ('data',))]

    # Misc configuration
    config.input.shuffle_buffer_size = 50_000
    config.log_training_steps = 50
    config.ckpt_steps = 1_000
    config.pp_modules = ['ops_general', 'ops_image', 'ops_text', 'proj.paligemma.ops']
    config.seed = 0
    config.wandb = not config.debug

    # Setup evaluation
    config_utils.setup_evaluation_config(config, config.res, prefix='', mode=config.mode, text_len=config.llm_text_len)

    # Debug mode settings
    if config.debug:
        config_utils.setup_debug_config(
            config,
            eval_only=config.debug_eval_only,
            eval_when_debugging=config.debug_eval_when_debugging,
            tiny_model=config.debug_tiny_model
        )

    return config