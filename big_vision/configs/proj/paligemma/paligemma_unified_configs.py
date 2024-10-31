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
        train_mode='pretrain',  # or 'finetune'
        loss_fn='softmax',
        dataset_name='laion400m/images', # laion400m/images, datacomp_recap/10M:1.0.0, datacomp_recap/50M:1.0.0, cambrian_dataset/10M:1.0.0,
        org_caption_ratio=1.0, # only for datacomp_recap
        
        # Model configs
        drop_path_rate=0.0,
        freeze_vit=False,
        vit_backbone=None,
        img_variant='B/16',
        img_beit_init=False,
        img_qknorm=False,
        freeze_llm=False,
        llm_variant='gemma_2b',
        llm_ckpt="partial_frozen:9", # partial_frozen:9, full
        llm_head='gap', # 'none', 'gap', 'map'
        llm_lr_mult=0.01, # 0.01, 0.1
        llm_dropout=0.0,
        llm_clean_vocab=True,
        llm_projection=True,
        
        # Training params
        lr=1e-3,
        wd=1e-4,
        epoch=-1.0,
        total_steps=-1,
        total_samples=-1,
        batch_size=16384,
        dtype='bfloat16', # 'float32', 'bfloat16'
        
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