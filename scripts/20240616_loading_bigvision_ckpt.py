import os
import sys
sys.path.append("/home/austinwang/austin_big_vision")
import big_vision.utils as u
from tensorflow.io import gfile

workdir = "gs://us-central2-storage/tensorflow_datasets/siglip_replication_pod_04-11_2247"
save_ckpt_path = os.path.join(workdir, "checkpoint.bv")

resume_ckpt_path = None
if save_ckpt_path and gfile.exists(f"{save_ckpt_path}-LAST"):
  resume_ckpt_path = save_ckpt_path

# running this in a notebook will lead to "RuntimeError: asyncio.run() cannot be called from a running event loop"
loaded = u.load_checkpoint_ts(resume_ckpt_path, tree=None, shardings=None)

print(loaded.keys())