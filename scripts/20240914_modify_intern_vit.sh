#!/bin/bash

FILE_PATH="/home/austinwang/.cache/huggingface/hub/models--OpenGVLab--InternViT-6B-224px/snapshots/1a1a6cb877785e21ee1a76ae26c757bb8bf68389/modeling_intern_vit.py"

sed -i '
/try:/,/has_flash_attn = False/c\
has_flash_attn = False\
print('\''FlashAttention is disabled for TPU compatibility.'\'')
' "$FILE_PATH"

echo "File has been modified to disable FlashAttention."
