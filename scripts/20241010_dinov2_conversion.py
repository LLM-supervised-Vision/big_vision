import numpy as np
import jax.numpy as jnp
from flax.training import checkpoints
from flax.core.frozen_dict import unfreeze
import os

def load_flax_checkpoint():
    from transformers import FlaxDinov2Model
    flax_model = FlaxDinov2Model.from_pretrained("facebook/dinov2-base")
    return flax_model._params

def convert_dinov2_to_bigvision(flax_params):
    bv_params = {}
    
    # Embedding layer
    bv_params['embedding/kernel'] = jnp.transpose(flax_params['embeddings']['patch_embeddings']['projection']['kernel'], (3, 2, 0, 1))
    bv_params['embedding/bias'] = flax_params['embeddings']['patch_embeddings']['projection']['bias']
    
    # Position embedding
    bv_params['pos_embedding'] = flax_params['embeddings']['position_embeddings']
    
    # Transformer blocks
    for i in range(12):  # Assuming 12 layers
        prefix = f'Transformer/encoderblock_{i}/'
        flax_prefix = f'encoder/layer/{i}/'
        
        # Layer Norm
        bv_params[prefix + 'LayerNorm_0/scale'] = flax_params[flax_prefix + 'norm1']['scale']
        bv_params[prefix + 'LayerNorm_0/bias'] = flax_params[flax_prefix + 'norm1']['bias']
        bv_params[prefix + 'LayerNorm_1/scale'] = flax_params[flax_prefix + 'norm2']['scale']
        bv_params[prefix + 'LayerNorm_1/bias'] = flax_params[flax_prefix + 'norm2']['bias']
        
        # Multi-Head Attention
        bv_params[prefix + 'MultiHeadDotProductAttention_0/query/kernel'] = flax_params[flax_prefix + 'attention/attention/query/kernel'].reshape(768, 12, 64)
        bv_params[prefix + 'MultiHeadDotProductAttention_0/query/bias'] = flax_params[flax_prefix + 'attention/attention/query/bias'].reshape(12, 64)
        bv_params[prefix + 'MultiHeadDotProductAttention_0/key/kernel'] = flax_params[flax_prefix + 'attention/attention/key/kernel'].reshape(768, 12, 64)
        bv_params[prefix + 'MultiHeadDotProductAttention_0/key/bias'] = flax_params[flax_prefix + 'attention/attention/key/bias'].reshape(12, 64)
        bv_params[prefix + 'MultiHeadDotProductAttention_0/value/kernel'] = flax_params[flax_prefix + 'attention/attention/value/kernel'].reshape(768, 12, 64)
        bv_params[prefix + 'MultiHeadDotProductAttention_0/value/bias'] = flax_params[flax_prefix + 'attention/attention/value/bias'].reshape(12, 64)
        bv_params[prefix + 'MultiHeadDotProductAttention_0/out/kernel'] = flax_params[flax_prefix + 'attention/output/dense/kernel'].reshape(12, 64, 768)
        bv_params[prefix + 'MultiHeadDotProductAttention_0/out/bias'] = flax_params[flax_prefix + 'attention/output/dense/bias']
        
        # MLP
        bv_params[prefix + 'MlpBlock_0/Dense_0/kernel'] = flax_params[flax_prefix + 'mlp/fc1/kernel']
        bv_params[prefix + 'MlpBlock_0/Dense_0/bias'] = flax_params[flax_prefix + 'mlp/fc1/bias']
        bv_params[prefix + 'MlpBlock_0/Dense_1/kernel'] = flax_params[flax_prefix + 'mlp/fc2/kernel']
        bv_params[prefix + 'MlpBlock_0/Dense_1/bias'] = flax_params[flax_prefix + 'mlp/fc2/bias']
    
    # Final Layer Norm
    bv_params['Transformer/encoder_norm/scale'] = flax_params['layernorm']['scale']
    bv_params['Transformer/encoder_norm/bias'] = flax_params['layernorm']['bias']
    
    return bv_params

def save_bigvision_checkpoint(params, output_path):
    np.savez(output_path, **params)

def main():
    output_path = '/home/austinwang/bigvision_dinov2.npz'
    
    print("Loading Flax checkpoint...")
    flax_params = load_flax_checkpoint()
    
    print("Converting parameters...")
    bv_params = convert_dinov2_to_bigvision(unfreeze(flax_params))
    
    print("Saving Big Vision compatible checkpoint...")
    save_bigvision_checkpoint(bv_params, output_path)
    
    print(f"Conversion complete. Checkpoint saved to {output_path}")

if __name__ == "__main__":
    main()