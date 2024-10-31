# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""pp ops."""

import functools
import string

from big_vision.pp import ops_text
from big_vision.pp import utils
from big_vision.pp.registry import Registry
import big_vision.pp.tokenizer as bv_tok
import numpy as np
import tensorflow as tf


@Registry.register('tokenizers.gemma')
def get_tokenizer_gemma(
    tokensets=(),
    model='gs://big_vision/gemma_tokenizer.model',
):
  # See (internal link) for colab playground.
  return ops_text.SentencepieceTokenizer(model=model, tokensets=tokensets)


@functools.cache
def tokenize_constant(model, text, bos='no', eos='no', length=None):
  """Tokenize a constant string, with memoization."""
  assert eos in ('no', 'yes', 'sticky')
  assert bos in ('no', 'yes')
  tokenizer = bv_tok.get_tokenizer(model)
  tokens = tokenizer.to_int(
      text, bos=bos == 'yes', eos=eos in ('yes', 'sticky'))

  if length is None:
    return tokens

  if len(tokens) > length:
    if eos == 'sticky':
      return np.r_[tokens[:length-1], tokens[-1]]
    else:
      return tokens[:length]
  else:
    return np.pad(tokens, [(0, length - len(tokens))],
                  constant_values=tokenizer.pad_token)


@Registry.register('preprocess_ops.tolen')
@utils.InKeyOutKey(indefault=None, outdefault=None, with_data=True)
def get_tolen(length, *, sticky_end=False, pad_value=None, pad_key=None):
  """Gets token to a fixed length."""
  def _tolen(x, data):
    if not length:
      return x

    xlen = tf.shape(x)[0]

    if sticky_end:
      trunc_fn = lambda: tf.concat([x[:length - 1], x[-1:]], axis=0)
    else:
      trunc_fn = lambda: x[:length]

    # Potentially get the pad value from a data key (to be tokenizer agnostic).
    pad_value_ = pad_value
    if pad_key:
      pad_value_ = data[pad_key]
      # If coming from a previous tokenization op, it's probably 1D; take first.
      if getattr(pad_value_, 'ndim', 0) == 1:
        pad_value_ = pad_value_[0]
    assert pad_value_ is not None, 'Need either pad_value or pad_key.'

    pad_fn = lambda: tf.pad(x, [(0, length - xlen)], constant_values=pad_value_)
    out = tf.cond(xlen >= length, trunc_fn, pad_fn)
    out.set_shape([length])
    return out
  return _tolen


@Registry.register('preprocess_ops.tok')
def get_tokenize(model, length=None, *, bos='no', eos='no',
                 text=None, key=None, inkey=None, outkey=None):
  """Tokenizes and optionally truncates/pads a string."""

  assert eos in ('no', 'yes', 'sticky')
  assert bos in ('no', 'yes')
  outkey_ = outkey or key
  inkey_ = inkey or key

  if text is not None:
    assert inkey is None, 'Either inkey or text, not both.'
    tokens = tokenize_constant(model, text, bos=bos, eos=eos, length=length)
    def _pp_tokenize_text(data):
      data[outkey_] = tokens
      return data
    return _pp_tokenize_text

  tokenizer = bv_tok.get_tokenizer(model)

  def _pp_tokenize(data):
    assert getattr(data[inkey_], 'ndim', 0) == 0, (
        f'Can only tokenize single string ({inkey_}, {data[inkey_].ndim}-D)')

    toks = tokenizer.to_int_tf_op(
        data[inkey_], bos=bos == 'yes', eos=eos in ('yes', 'sticky'))

    tolen = get_tolen(
        length, sticky_end=eos == 'sticky',
        pad_value=bv_tok.get_tokenizer(model).pad_token,
        key='tmp',
    )
    toks = tolen({'tmp': toks})['tmp']

    data[outkey_] = toks
    return data
  return _pp_tokenize


@Registry.register('preprocess_ops.masked_concat')
def get_masked_concat(keys, outkey='text', **masks):
  assert all(len(keys) == len(m) for m in masks.values()), (keys, masks)
  def _masked_concat(data):
    data[outkey] = tf.concat([data[k] for k in keys], axis=0)
    for mask_name, mask_vals in masks.items():
      m = [tf.fill(tf.shape(data[k]), v) for k, v in zip(keys, mask_vals)]
      data[mask_name] = tf.concat(m, axis=0)
    return data
  return _masked_concat


@Registry.register('preprocess_ops.strfmt')
def get_strfmt(template, outkey='text'):
  """Formats a string template with content form the data dict."""

  def _template(data):
    outputs = []
    parts = string.Formatter().parse(template)
    for (literal_text, field_name, format_spec, conversion) in parts:
      # For now, we keep it simple and don't support fancy format specs.
      # But we can add support to that via py_func as soon as we need it.
      assert not format_spec and not conversion
      outputs.append(tf.constant(literal_text))
      if field_name:
        value = data[field_name]
        # Convert any non-strings (numbers, vectors) to a string.
        if tf.convert_to_tensor(value).dtype != tf.string:
          value = tf.strings.format('{}', value, summarize=-1)
        outputs.append(value)
    data[outkey] = tf.strings.join(outputs)
    return data

  return _template


@Registry.register('preprocess_ops.strjoin')
@utils.InKeyOutKey()
def get_strjoin(glue):
  def _strjoin(x):
    return tf.strings.reduce_join(x, separator=glue)
  return _strjoin


@Registry.register('preprocess_ops.majority')
@utils.InKeyOutKey()
def get_majority():
  def _majority(x):
    val, _, count = tf.unique_with_counts(x)  # Sadly, stablesorted.
    return val[tf.argmax(count)]
  return _majority


@Registry.register('preprocess_ops.getidx')
def getidx(inkey, index_key, outkey=None):
  """Indexes a tensor and stores result in outkey."""
  def _getidx(data):
    idx = data[index_key]
    array = data[inkey]
    data[outkey or inkey] = array[idx]
    return data
  return _getidx


@Registry.register("preprocess_ops.language_only_filtering")
def get_language_only_filtering():
  def _language_only_filtering(data):
    has_image = data.get('has_image', True)
    if not has_image:
      data['has_image'] = False
    return data
  return _language_only_filtering

@Registry.register("preprocess_ops.process_conversations")
def get_process_conversations():
  def _process_conversations(data):
    from_tensor = data['conversations']['from']
    value_tensor = data['conversations']['value']
    
    prefixes = tf.TensorArray(tf.string, size=0, dynamic_size=True)
    suffixes = tf.TensorArray(tf.string, size=0, dynamic_size=True)
    
    def clean_tok_text(text):
      # Remove '<image>' and '\n'
      cleaned = tf.strings.strip(
        tf.strings.regex_replace(text, '<image>', '')
        # tf.strings.regex_replace(text, '<image>|\\n', '')
      )
      return cleaned
    
    for i in tf.range(tf.shape(from_tensor)[0]):
      cleaned = clean_tok_text(value_tensor[i])
      if from_tensor[i] == b'human':
          prefixes = prefixes.write(prefixes.size(), cleaned)
      elif from_tensor[i] == b'gpt':
          suffixes = suffixes.write(suffixes.size(), cleaned)
    
    data['prefixes'] = prefixes.stack()
    data['suffixes'] = suffixes.stack()
    
    return data
  
  return _process_conversations

@Registry.register("preprocess_ops.tok_multi")
def get_tokenize_multi(model, *, key=None, inkey=None, outkey=None):
    """Tokenizes multiple sentences without padding or truncation."""
    outkey_ = outkey or key
    inkey_ = inkey or key

    tokenizer = bv_tok.get_tokenizer(model)

    def _pp_tokenize_multi(data):
        sentences = data[inkey_]
        
        def tokenize_single(sentence):
            tokens = tokenizer.to_int_tf_op(sentence, bos=False, eos=False)
            return tokens

        tokenized = tf.map_fn(
            tokenize_single,
            sentences,
            fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.int32, ragged_rank=0)
        )
        data[outkey_] = tokenized
        data['bos_token'] = tf.convert_to_tensor([tokenizer.bos_token], dtype=tf.int32)
        data['eos_token'] = tf.convert_to_tensor([tokenizer.eos_token], dtype=tf.int32)
        data['sep_token'] = tf.convert_to_tensor(tokenizer.to_int_tf_op('\n', bos=False, eos=False), dtype=tf.int32)
        return data

    return _pp_tokenize_multi

@Registry.register("preprocess_ops.masked_concat_multi")
def get_masked_concat_multi(keys):
    def _masked_concat_multi(data):
        prefixes = data[keys[0]]
        suffixes = data[keys[1]]
        
        bos_token = data['bos_token']
        eos_token = data['eos_token']
        sep_token = data['sep_token']
        
        def interleave_conversations(inputs):
            prefix, suffix = inputs
            # For each prefix-suffix pair, we create sequence and corresponding mask
            interleaved = tf.concat([prefix, sep_token, suffix, sep_token], axis=0)
            # Generate mask: 0 for prefix and its sep_token, 1 for suffix and its sep_token
            mask_prefix = tf.zeros_like(prefix, dtype=tf.int32)
            mask_sep1 = tf.zeros_like(sep_token, dtype=tf.int32)
            mask_suffix = tf.ones_like(suffix, dtype=tf.int32)
            mask_sep2 = tf.ones_like(sep_token, dtype=tf.int32)
            mask = tf.concat([mask_prefix, mask_sep1, mask_suffix, mask_sep2], axis=0)
            return interleaved, mask
        
        # Process each prefix-suffix pair
        results = tf.map_fn(
            interleave_conversations,
            (prefixes, suffixes),
            fn_output_signature=(
                tf.RaggedTensorSpec(shape=[None], dtype=tf.int32, ragged_rank=0),
                tf.RaggedTensorSpec(shape=[None], dtype=tf.int32, ragged_rank=0)
            )
        )
        
        # Separate interleaved text and masks
        interleaved, masks = results
        
        # Flatten the interleaved conversations and masks
        flattened = interleaved.merge_dims(0, 1)
        flattened_mask = masks.merge_dims(0, 1)
        
        # Add BOS and EOS tokens to text and corresponding masks
        final_text = tf.concat([bos_token, flattened, eos_token], axis=0)
        mask_ar = tf.concat([
            tf.zeros_like(bos_token, dtype=tf.int32),  # mask for BOS
            flattened_mask,                            # mask for interleaved content
            tf.ones_like(eos_token, dtype=tf.int32)    # mask for EOS
        ], axis=0)
        
        # Use the same mask for loss
        mask_loss = mask_ar
        
        data['text'] = final_text
        data['mask_ar'] = mask_ar
        data['mask_loss'] = mask_loss
        
        return data
    
    return _masked_concat_multi

import tensorflow as tf
from PIL import Image
import io
import numpy as np

@Registry.register("preprocess_ops.cambrian_image_pp")
@utils.InKeyOutKey()
def get_cambrian_image_pp(size=224, value_range=(-1, 1)):
    """
    Combined preprocessing function for Cambrian dataset images.
    Handles scalar string input containing image bytes and gracefully handles decoding errors.
    """
    size = utils.maybe_repeat(size, 2)

    def _cambrian_image_pp(image_bytes):
        # tf.print("Input shape:", tf.shape(image_bytes))
        # tf.print("Input dtype:", image_bytes.dtype)

        def process_single_image(bytes_data):
            # tf.print("Processing single image, bytes length:", tf.strings.length(bytes_data))
            
            # Decode image
            try:
                image = tf.io.decode_image(bytes_data, channels=3, expand_animations=False)
                # tf.print("Decoded image shape:", tf.shape(image))
            except tf.errors.InvalidArgumentError as e:
                try:
                    # pil_image = Image.open(io.BytesIO(bytes_data))
                    pil_image = Image.open(bytes_data.numpy())
                    pil_image = pil_image.convert('RGB')
                    image = tf.convert_to_tensor(np.array(pil_image))
                except Exception as e:
                    tf.print(f"Error decoding image: {e}")
                    exit()
                    # return tf.zeros([size[0], size[1], 3], dtype=tf.float32)

            # Resize image
            image = tf.image.resize(image, size, method='bilinear', antialias=True)

            # Convert to float and scale to desired value range
            image = tf.cast(image, tf.float32)
            image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
            image = image * (value_range[1] - value_range[0]) + value_range[0]

            return image

        # Use tf.py_function to wrap the process_single_image function
        def wrapped_process_single_image(bytes_data):
            return tf.py_function(process_single_image, [bytes_data], tf.float32)

        result = tf.cond(
            tf.equal(tf.strings.length(image_bytes), 0),
            lambda: tf.zeros([size[0], size[1], 3], dtype=tf.float32),
            lambda: wrapped_process_single_image(image_bytes)
        )
        
        # Ensure the result has the correct shape
        result = tf.ensure_shape(result, [size[0], size[1], 3])
        
        # tf.print("Output shape:", tf.shape(result))
        return result

    return _cambrian_image_pp