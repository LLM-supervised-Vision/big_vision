import os
import tensorflow as tf
import requests
from datasets import load_dataset
import itertools
from tqdm import tqdm
from io import BytesIO
from PIL import Image

def download_image(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')  # Convert to RGB to ensure consistency
        width, height = img.size
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='JPEG')
        return img_byte_arr.getvalue(), width, height
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None, None, None

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_example(sample):
    image_data, width, height = download_image(sample['url'])
    if image_data is None:
        return None
    
    feature = {
        'image': _bytes_feature(image_data),
        'url': _bytes_feature(sample['url'].encode('utf-8')),
        're_caption': _bytes_feature(sample['re_caption'].encode('utf-8')),
        'org_caption': _bytes_feature(sample['org_caption'].encode('utf-8')),
        'width': _int64_feature(width),
        'height': _int64_feature(height),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def convert_to_tfrecord(dataset, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        total, success = 0, 0
        for sample in tqdm(dataset):
            total += 1
            tf_example = create_example(sample)
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())
                success += 1
        print(f"Converted {success}/{total} samples ({success/total:.2%}), saved to {output_file}")

# Load the dataset
ds = load_dataset("UCSC-VLAA/Recap-DataComp-1B", split="train", streaming=True)

# Take the first n samples
limited_ds = itertools.islice(ds, 100)

# Convert the limited dataset to TFRecord
out_dir = "/home/austinwang/tensorflow_datasets/downloads"
out_path = os.path.join(out_dir, "datacomp_recap_train.tfrecord")
if not os.path.exists(out_dir): os.makedirs(out_dir) 
convert_to_tfrecord(limited_ds, out_path)


