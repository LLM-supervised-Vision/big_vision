"""Configuration utilities for PaLiGeMMA training."""

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

# Dataset sizes for step calculation
DATASET_SIZES = {
    'laion400m/images': 379_600_897,
    'datacomp_recap/10M': 8_344_225,
    'datacomp_recap/50M': 41_598_460,
    'cambrian_dataset/10M': 9_784_414,
}

def get_text_length(config):
    """Determine text length based on dataset and settings."""
    dataset_type = config.dataset_name.split("/")[0]
    if dataset_type == 'cambrian_dataset': 
        return 320
    elif dataset_type == 'datacomp_recap' and config.org_caption_ratio < 1.0: 
        return 128
    else: 
        return 64 # laion400m or datacomp_recap with org_caption=1.0

def create_training_data_config(config, prefix=""):
    """Creates training data configuration."""
    if not isinstance(config.res, int) or config.res <= 0:
        raise ValueError(f"Resolution must be a positive integer, got {config.res}")
    if not isinstance(config.org_caption_ratio, float) or not 0 <= config.org_caption_ratio <= 1:
        raise ValueError(f"Original caption ratio must be between 0 and 1, got {config.org_caption_ratio}")
    
    input_config = bvcc.parse_arg('')
    input_config.data = {
        'name': config.dataset_name,
        'split': 'train',
        'data_dir': 'gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets'
    }
    
    dataset_type = config.dataset_name.split("/")[0]
    
    preprocessing_ops = {
        'laion400m': [
            f'decode|resize({config.res})|value_range(-1,1)',
            f'strfmt({prefix}, outkey="prefix")',
            'copy(inkey="caption", outkey="suffix")',
            combine_and_keep_train(config.llm_text_len),
        ],
        'datacomp_recap': [
            f'decode|resize({config.res})|value_range(-1,1)',
            f'strfmt({prefix}, outkey="prefix")',
            f'ratio_choice(inkey=["org_caption", "re_caption"], '
            f'outkey="caption", ratios=[{config.org_caption_ratio}, {1-config.org_caption_ratio}])|'
            f'copy(inkey="caption", outkey="suffix")',
            combine_and_keep_train(config.llm_text_len),
        ],
        'cambrian_dataset': [
            f'decode|resize({config.res})|value_range(-1,1)',
            cambrian_pp(config.llm_text_len),
        ]
    }
    
    if dataset_type not in preprocessing_ops:
        raise ValueError(f"Unknown dataset_name: {dataset_type}")
    
    input_config.pp = '|'.join(preprocessing_ops[dataset_type])
    if dataset_type == 'cambrian_dataset':
        input_config['data_dir'] = 'gs://us-central2-storage/tensorflow_datasets'
    input_config.batch_size = config.batch_size
    return input_config

def calculate_total_steps(config):
    """Calculate total steps based on epoch, total_steps, or total_samples."""
    if sum(x >= 0 for x in [config.epoch, config.total_samples, config.total_steps]) > 1:
        raise ValueError("Only one of epoch, total_samples, or total_steps can be specified")

    if config.total_steps >= 0:
        return config.total_steps
        
    if config.epoch >= 0:
        dataset_base = config.dataset_name.split(":")[0]
        if dataset_base not in DATASET_SIZES:
            raise ValueError(f"Unknown dataset: {dataset_base}")
        return int(DATASET_SIZES[dataset_base] * config.epoch / config.input.batch_size)
        
    if config.total_samples >= 0:
        return int(config.total_samples * 1e9 / config.input.batch_size)
        
    raise ValueError("Must specify either total_steps, epoch, or total_samples")


def setup_model_config(config):
    """Setup model configuration, initialization, and training schedule.
    
    This function handles:
    1. Training mode and learning rate setup
    2. ViT initialization based on training mode and backbone choice
    3. LLM configuration and checkpoint loading
    4. Training schedule configuration
    
    Args:
        config: Configuration object containing model settings
    """

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
    
    # Validate training mode
    if config.train_mode not in ['pretrain', 'finetune']:
        raise ValueError(f"train_mode must be 'pretrain' or 'finetune', got {config.train_mode}")
    
    # Set learning rate multipliers based on training mode
    vit_lr_mult = 1.0 if config.train_mode == 'pretrain' else 0.1
    config.lr_mults = [
        ('img/.*', vit_lr_mult),
        ('llm/.*', config.llm_lr_mult),
        ('t', 1.0),
    ]
    
    # Initialize model loading configuration
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

    # Handle ViT initialization based on training mode
    if config.train_mode == 'pretrain':
        if config.vit_backbone is not None:
            raise ValueError("vit_backbone should be None for pretraining (random initialization)")
        config.model_init = {'img': None, 'llm': None}  # LLM init will be set later
    else:  # finetune
        if config.vit_backbone not in ['gemma_supervised', 'clip+llm']:
            raise ValueError("For finetuning, vit_backbone must be either 'gemma_supervised' or 'clip+llm'")
            
        # Set up ViT backbone
        if config.vit_backbone == 'gemma_supervised':
            backbone = ("gs://us-central2-storage/tensorflow_datasets/mllm_ckpts/paligemma/"
                       "gemma2b-partial_frozen99-0.01-gap_b16-F_contrastive_bs16k_s3b_lr1e-3_wd1e-4_bf16_09-01_0446")
            ckpt_cfg_path = f'{backbone}/config.json'
            ckpt_cfg = ml_collections.ConfigDict(json.load(tf.io.gfile.GFile(ckpt_cfg_path, 'r')))
            config.model_init = f"{backbone}/checkpoint.bv-{ckpt_cfg.total_steps:09d}"
            config.model = ckpt_cfg.model
        else:  # clip+llm
            backbone = ("gs://us-central2-storage/tensorflow_datasets/vit-b-16_3b_pretraining/"
                       "clip_bs16384_warm0.03_lr1e-3_wd1e-4_bf16_qknorm-F_b2-0.95_12lyr_07-25_1415")
            ckpt_cfg_path = f'{backbone}/config.json'
            ckpt_cfg = ml_collections.ConfigDict(json.load(tf.io.gfile.GFile(ckpt_cfg_path, 'r')))
            config.model_init = {'img': f"{backbone}/checkpoint.bv-{ckpt_cfg.total_steps:09d}:img", 'llm': None}
            config.model.img = ckpt_cfg.model.image
            config.model.img['pool_type'] = 'none'
            config.model_load['img_load_kw'] = {'dont_load': ['head/.*']}

    # Handle LLM configurations
    llm_ckpt = 'gs://us-central2-storage/tensorflow_datasets/gemma2b.npz'
    config.schedule_base = {'decay_type': 'cosine', 'warmup_percent': config.warmup_percent}
    config.schedule = [
        ('img/.*', None if config.freeze_vit else config.schedule_base),
        ('llm/.*', None if config.freeze_llm else config.schedule_base),
        ('t', config.schedule_base),
    ]

    if ':' in config.llm_ckpt:
        base_type, lyrs_frozen = config.llm_ckpt.split(':')
        config.llm_ckpt = base_type
        
        if base_type in ('partial_frozen', 'scratch-partial_frozen'):
            config.model.llm['lyrs_frozen'] = int(lyrs_frozen)
            if not config.freeze_llm:
                config.schedule = [
                    ('img/.*', None if config.freeze_vit else config.schedule_base),
                    ('llm/layers/frozen/.*', None),
                    ('.*', config.schedule_base),
                ]

    # Process different LLM checkpoint types
    match config.llm_ckpt:
        case 'full':
            pass
        
        case 'half':
            config.model.llm['variant'] = 'gemma_2b_half'
            llm_ckpt = '/home/austinwang/gemma2b_half.npz'
            config.model_load['llm_load_kw']['dont_load'].append('final_norm/scale')
        
        case '6lyr':
            config.model.llm['variant'] = 'gemma_6lyr'
            llm_ckpt = '/home/austinwang/gemma2b_first_6.npz'
            config.model_load['llm_load_kw']['dont_load'].append('final_norm/scale')
        
        case 'partial_frozen':
            assert not config.freeze_llm, "partial_frozen is for unfreezing"
            config.schedule = [
                ('img/.*', None if config.freeze_vit else config.schedule_base),
                ('llm/layers/frozen/.*', None),
                ('.*', config.schedule_base),
            ]
        
        case 'ln':
            assert not config.freeze_llm, "ln is for unfreezing"
            config.schedule = [
                ('img/.*', None if config.freeze_vit else config.schedule_base),
                ('.*norm/.*', config.schedule_base),
                ('.*', None),
            ]
        
        case 'scratch':
            llm_ckpt = None
            config.model_init = None
            config.model_load = {}
        
        case 'scratch-partial_frozen':
            llm_ckpt = None
            config.model_init = None
            config.model_load = {}
            assert not config.freeze_llm, "scratch-partial_frozen is for unfreezing"
            config.schedule = [
                ('img/.*', None if config.freeze_vit else config.schedule_base),
                ('llm/layers/frozen/.*', None),
                ('.*', config.schedule_base),
            ]
        
        case 'adapter':
            assert config.llm_head == 'ffn', "adapter is for ffn head"
            assert not config.llm_projection, "adapter is for ffn head"
            config.schedule = [
                ('img/.*', None if config.freeze_vit else config.schedule_base),
                ('.*Adapter/.*', config.schedule_base),
                ('t', config.schedule_base),
                ('.*', None),
            ]
        
        case _:
            raise ValueError(f"Unknown llm_ckpt: {config.llm_ckpt}")

    # Set model initialization if llm_ckpt is provided
    if llm_ckpt is not None and isinstance(config.model_init, ConfigDict):
        config.model_init['llm'] = llm_ckpt
            
    return config


def setup_evaluation_config(config, prefix=""):
    """Sets up evaluation configuration based on training mode."""
    if config.mode not in ['contrastive', 'generative']:
        raise ValueError(f"mode must be 'contrastive' or 'generative', got {config.mode}")
        
    if config.mode == "contrastive":
        config.evals = {}  # Reset evals
        config.evals.retrieval_coco = common.get_coco(
            pred='contrastive_logits',
            pp_img=f'resize({config.res})|value_range(-1, 1)',
            pp_txt='|'.join([
                f'strfmt({prefix}, outkey="prefix")',
                'copy(inkey="texts", outkey="suffix")',
                combine_and_keep_eval(config.llm_text_len, eos='yes'),
                'copy(inkey="text", outkey="labels")',
            ]),
            log_steps=1000
        )
        
        config.evals.zeroshot_imagenet = common.get_disclf(
            pred='contrastive_logits',
            sz=config.res,
            pp_txt='|'.join([
                f'strfmt({prefix}, outkey="prefix")',
                'copy(inkey="texts", outkey="suffix")',
                combine_and_keep_eval(config.llm_text_len, eos='yes'),
                'copy(inkey="text", outkey="labels")',
            ]),
            dataset_names=('imagenet2012', 'imagenet_v2', 'imagenet2012_real'),
            log_steps=1000
        )
        
    else:  # generative
        config.evals = {}
        # pp = '|'.join([
        #     f'strfmt({prefix}, outkey="prefix")',
        #     'copy(inkey="label", outkey="suffix")',
        #     combine_and_keep_eval(config.llm_text_len, keep=('text', 'mask_ar')),
        #     'copy(inkey="text", outkey="labels")',
        # ])
        # config.evals['imagenet/scoring'] = {
        #     'type': 'proj.paligemma.scoring_classifier',
        #     'pred': 'score',
        #     'log_percent': 0.1,
        #     'data': {'name': 'imagenet2012', 'split': 'validation[:320]'},
        #     'pp_fn': f'decode|resize({config.res})|keep("image", "label")',
        #     'pp_txt': pp,
        # }
    
    return config


def setup_debug_config(config):
    """Setup debug configuration with flexible options."""
    config.wandb = False
    
    if config.debug_eval_only:
        config.total_steps = 0
        config.lr = 0.0
        config.wd = 0.0
    else:
        config.input.shuffle_buffer_size = None
        config.input.batch_size = 32
        config.total_steps = 10
        config.epoch = -1.0
        config.total_samples = -1
        config.schedule = [('.*', dict(decay_type='cosine', warmup_steps=3))]
        config.log_training_steps = 1
    
    if not config.debug_eval_when_debugging:
        config.evals = {}
    else:
        for k in config.evals:
            config.evals[k]['batch_size'] = 32
    
    if config.debug_tiny_model:
        config.model.img = dict(variant='mu/16', pool_type='none')
        config.model.llm = dict(variant='gemma_debug')
        config.model_init = None
        config.model_load = {}

    return config
