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
        debug_eval_only=False,
        debug_eval_when_debugging=False,
        debug_tiny_model=False,
    )

    # Setup input pipeline
    config.llm_text_len = config_utils.get_text_length(config)
    config.input = config_utils.create_training_data_config(config, prefix='')
    
    # Optimizer configuration
    config.optax_name = 'scale_by_adam'
    config.optax = {'b1': 0.9, 'b2': 0.95}
    config.lr = config.lr
    config.wd = config.wd
    config.grad_clip_norm = 1.0
    config.label_smoothing = 0.0
    config.total_steps = config_utils.calculate_total_steps(config)

    # Setup model initialization and loading
    config = config_utils.setup_model_config(config)

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
    config = config_utils.setup_evaluation_config(config, prefix='')

    # Debug mode settings
    if config.debug:
        config = config_utils.setup_debug_config(config)

    return config