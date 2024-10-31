"""Configuration for PaLiGeMMA training."""

import os
import logging
import json
import tensorflow as tf
import ml_collections
from ml_collections import ConfigDict

import big_vision.configs.common as bvcc
from big_vision.configs.proj.image_text import common
from big_vision.configs.proj.paligemma.transfers.common import (
    combine_and_keep_train,
    combine_and_keep_eval,
    cambrian_pp,
    TOKENIZER
)

def create_training_data_config(res, prefix, text_len=64, dataset_name='laion400m/images', org_caption_ratio=0.5):
    """Creates training data configuration.
    
    Args:
        res: Image resolution
        prefix: Text prefix for prompts
        text_len: Maximum text length
        dataset_name: Name of the dataset
        org_caption_ratio: Ratio of original captions to use (for datacomp_recap)
    """
    config = bvcc.parse_arg('')
    config.data = {
        'name': dataset_name,
        'split': 'train',
        'data_dir': 'gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets'
    }
    
    dataset_type = dataset_name.split("/")[0]
    
    preprocessing_ops = {
        'laion400m': [
            f'decode|resize({res})|value_range(-1,1)',
            f'strfmt("{prefix}", outkey="prefix")',
            'copy(inkey="caption", outkey="suffix")',
            combine_and_keep_train(text_len),
        ],
        'datacomp_recap': [
            f'decode|resize({res})|value_range(-1,1)',
            f'strfmt("{prefix}", outkey="prefix")',
            f'ratio_choice(inkey=["org_caption", "re_caption"], '
            f'outkey="caption", ratios=[{org_caption_ratio}, {1-org_caption_ratio}])|'
            f'copy(inkey="caption", outkey="suffix")',
            combine_and_keep_train(text_len),
        ],
        'cambrian_dataset': [
            f'decode|resize({res})|value_range(-1,1)',
            cambrian_pp(text_len),
        ]
    }
    
    if dataset_type not in preprocessing_ops:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
        
    if dataset_type == 'cambrian_dataset':
        config.data['data_dir'] = 'gs://us-central2-storage/tensorflow_datasets'
        
    config.pp = '|'.join(preprocessing_ops[dataset_type])
    
    return config

def setup_evaluation_config(config, res, text_len=64, prefix='', mode='contrastive', **kwargs):
    """Sets up evaluation configuration based on training mode."""
    if mode == "contrastive":
        config.evals.retrieval_coco = common.get_coco(
            pred='contrastive_logits',
            pp_img=f'resize({res})|value_range(-1, 1)',
            pp_txt='|'.join([
                f'strfmt("{prefix}", outkey="prefix")',
                'copy(inkey="texts", outkey="suffix")',
                combine_and_keep_eval(text_len, eos='yes'),
                f'copy(inkey="text", outkey="labels")',
            ]),
            log_steps=1000,
            **kwargs
        )
        
        config.evals.zeroshot_imagenet = common.get_disclf(
            pred='contrastive_logits',
            sz=res,
            pp_txt='|'.join([
                f'strfmt("{prefix}", outkey="prefix")',
                'copy(inkey="texts", outkey="suffix")',
                combine_and_keep_eval(text_len, eos='yes'),
                f'copy(inkey="text", outkey="labels")',
            ]),
            dataset_names=('imagenet2012', 'imagenet_v2', 'imagenet2012_real'),
            log_steps=1000,
            **kwargs
        )
        
    elif mode == "generative":
        config.evals = {}
        pp = '|'.join([
            f'strfmt("{prefix}", outkey="prefix")',
            'copy(inkey="label", outkey="suffix")',
            combine_and_keep_eval(text_len, keep=('text', 'mask_ar')),
            f'copy(inkey="text", outkey="labels")',
        ])
        config.evals['imagenet/scoring'] = {
            'type': 'proj.paligemma.scoring_classifier',
            'pred': 'score',
            'log_percent': 0.1,
            'data': {'name': 'imagenet2012', 'split': 'validation[:320]'},
            'pp_fn': f'decode|resize({res})|keep("image", "label")',
            'pp_txt': pp,
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

def setup_model_config(config):
    """Sets up model configuration including initialization and loading."""
    dont_load = []
    if config.model.llm['head'] == 'map':
        dont_load.append('MAPHead.*')
    if config.model.llm['head'] == 'ffn':
        dont_load.append('FFNAdapter.*')
    if config.model.llm['projection']:
        dont_load.append('head/.*')
        
    config.model_load = {
        'img_load_kw': {},
        'llm_load_kw': {'dont_load': dont_load}
    }
    
    llm_ckpt_configs = {
        'full': {},
        'half': {
            'variant': 'gemma_2b_half',
            'ckpt_path': '/home/austinwang/gemma2b_half.npz',
            'extra_dont_load': ['final_norm/scale']
        },
        '6lyr': {
            'variant': 'gemma_6lyr',
            'ckpt_path': '/home/austinwang/gemma2b_first_6.npz',
            'extra_dont_load': ['final_norm/scale']
        },
        'adapter': {
            'schedule_override': [
                ('img/.*', None if config.freeze_vit else config.schedule_base),
                ('.*Adapter/.*', config.schedule_base),
                ('t', config.schedule_base),
                ('.*', None),
            ]
        }
    }
    
    # Handle special frozen cases
    if ':' in config.llm_ckpt:
        base_type, lyrs_frozen = config.llm_ckpt.split(':')
        if base_type in ('partial_frozen', 'scratch-partial_frozen'):
            config.model.llm['lyrs_frozen'] = int(lyrs_frozen)
            if not config.freeze_llm:
                config.schedule = [
                    ('img/.*', None if config.freeze_vit else config.schedule_base),
                    ('llm/layers/frozen/.*', None),
                    ('.*', config.schedule_base),
                ]
    
    # Apply config based on llm_ckpt type
    ckpt_type = config.llm_ckpt.split(':')[0] if ':' in config.llm_ckpt else config.llm_ckpt
    if ckpt_type in llm_ckpt_configs:
        cfg = llm_ckpt_configs[ckpt_type]
        if 'variant' in cfg:
            config.model.llm['variant'] = cfg['variant']
        if 'ckpt_path' in cfg:
            config.llm_ckpt_path = cfg['ckpt_path']
            if 'extra_dont_load' in cfg:
                config.model_load['llm_load_kw']['dont_load'].extend(cfg['extra_dont_load'])
        if 'schedule_override' in cfg:
            config.schedule = cfg['schedule_override']
    elif ckpt_type == 'scratch':
        config.model_init = None
        config.model_load = {}

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
        epoch=5.0,
        
        # Vision model params
        freeze_vit=False,
        img_variant='B/16',
        img_beit_init=False,
        img_qknorm=False,
        
        # Language model params
        freeze_llm=True,
        llm_variant='gemma_2b',
        llm_ckpt="full",
        llm_head='none',
        llm_lr_mult=0.1,
        llm_dropout=0.0,
        llm_clean_vocab=False,
        llm_projection=False,
        llm_text_len=64,
        
        # Training params
        batch_size=8192,
        total_samples=3.0,
        dtype='float32',
        debug=False,
    )

    # Setup input pipeline
    config.input = create_training_data_config(
        config.res,
        prefix='',
        text_len=config.llm_text_len,
        dataset_name=config.dataset_name,
        org_caption_ratio=config.org_caption_ratio
    )

    # Training parameters
    config.input.batch_size = config.batch_size
    config.total_steps = int(config.total_samples * 1e9 / config.input.batch_size)
    
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
    
    if not config.freeze_vit and not config.freeze_llm:
        config.lr_mults = [
            ('img/.*', 1.0),
            ('llm/.*', config.llm_lr_mult),
            ('t', 1.0),
        ]

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
    setup_model_config(config)

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
    setup_evaluation_config(config, config.res, prefix='', mode=config.mode, text_len=config.llm_text_len)

    # Handle dataset-specific configurations
    if config.dataset_name.split("/")[0] in ['datacomp_recap', 'cambrian_dataset']:
        dataset_sizes = {
            'datacomp_recap/10M': 8_344_225,
            'datacomp_recap/50M': 41_598_460,
            'cambrian_dataset/10M': 9_784_414,
        }
        
        dataset_base = config.dataset_name.split(":")[0]
        if dataset_base not in dataset_sizes:
            raise ValueError(f"Unknown dataset: {dataset_base}")
            
        config.total_steps = int(dataset_sizes[dataset_base] * config.epoch / config.input.batch_size)

    # Debug mode settings
    if config.debug:
        config.wandb = False
        config.input.shuffle_buffer_size = None
        config.input.batch_size = 32
        config.total_steps = 10
        config.schedule = [('.*', dict(decay_type='cosine', warmup_steps=3))]
        config.log_training_steps = 1
        config.evals = {}
        
        # Use tiny model for debugging
        config.model.img = dict(variant='mu/16', pool_type='none')
        config.model.llm = dict(variant='gemma_debug')
        config.model_init = None
        config.model_load = {}

    return config