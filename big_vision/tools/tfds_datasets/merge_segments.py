"""
This script merges the segments of selected versions of the dataset into 1.0.0 version.
1. worker_id=0 will read dataset_info.json files of all versions until final_version_id;
    and calculate the total number of bytes and shard lengths; 
    and update the dataset_info.json in the first folder.
2. all workers will move the files to the destination directory with the modified names.
    modified names: 
        laion400m-train.tfrecord-{new_file_index:05d}-of-{total_num_files:05d}
        new_file_index = file_index + calc_total_num_files(version=version_id-1)
        total_num_files = calc_total_num_files(version=final_version_id)
3. worker_id=0 will check the metadata in dataset_info.json in the first folder.
"""
import json
import logging
import threading
from absl import app
from absl import flags
from google.cloud import storage

FLAGS = flags.FLAGS
flags.DEFINE_string('destination_dir', None, 'Destination directory path')
flags.DEFINE_integer('final_version_id', None, 'Final version id')
flags.DEFINE_integer('num_workers', None, 'Number of workers')
flags.DEFINE_integer('worker_id', None, 'Worker id')

file_operation_lock = threading.Lock()

def move_and_rename_files(destination_dir, final_version_id, num_workers, worker_id):
    # read dataset_info.json files of all versions until final_version_id
    # calculate the total number of bytes and shard lengths
    # update the dataset_info.json in the 1.0.0 folder
    # return a list of accumulated number of files in each version
    # move the files to the destination directory with the modified names

    # Get the bucket
    client = storage.Client()
    bucket_name = destination_dir.split("/")[2] # us-central2-storage
    # logging.info(f"bucket_name: {bucket_name}")
    bucket = client.get_bucket(bucket_name)
    # logging.info(f"bucket: {bucket}")

    # Get blobs in the source directory
    prefix = "/".join(destination_dir.split("/")[3:]) # tensorflow_datasets/laion400m/images/1.0.0/
    # logging.info(f"prefix: {prefix}")
    blobs = list(bucket.list_blobs(prefix=prefix[:-2]))
    # logging.info(f"len(blobs): {len(blobs)}")
    
    # exclude blobs from 1.0.0 folder and blobs after final_version_id
    blobs = [blob for blob in blobs if blob.name.split("/")[3].split(".")[2]!="0"] 
    # logging.info(f"len(blobs): {len(blobs)}")
    blobs = [blob for blob in blobs if int(blob.name.split("/")[3].split(".")[2].split("_")[0])<=final_version_id]
    # logging.info(f"{worker_id}: len(blobs): {len(blobs)}")

    # Custom sorting key function to sort blobs based on the numeric value after the last dot
    def sort_key(blob):
        file_name = blob.name.split("/")[-2]
        version_number = file_name.split(".")[-1]  # Get the part after the last dot
        version_number = version_number.split("_")[0] # consider the case of 1.0.0_source folder, which is distinguished from 1.0.0 as destination folder
        return int(version_number)

    # Sort blobs based on the custom sorting key
    blobs_sorted = sorted(blobs, key=sort_key)
    logging.info(f"{worker_id}: len(blobs_sorted): {len(blobs_sorted)}")

    total_num_bytes = 0
    shard_lengths = []
    # stored as {version_id+1: accumulated number of files} {0:0, 1:28, 2:60, 3:None, 4:89, ...} (not real examples, just for explanation)
    # e.g. shard_lengths_accumulated[1] = 28 means there are 28 files before version 1 (not including version 1)
    shard_lengths_accumulated = {} 
    next_version_id_dict = {} # {0:1, 1:2, 2:4, 3:5, ...}
    previous_version_id = None

    # Read numBytes from dataset_info.json in each folder and add to total_num_bytes
    for blob in blobs_sorted:
        if blob.name.split("/")[-1] == "dataset_info.json":
            # logging.info(f"blob.name: {blob.name}")
            dataset_info_blob = bucket.get_blob(f"{blob.name}")
            dataset_info = json.loads(dataset_info_blob.download_as_string())
            total_num_bytes += int(dataset_info["splits"][0]["numBytes"])
            version_id = int(blob.name.split("/")[3].split(".")[2].split("_")[0])
            shard_lengths_accumulated[version_id] = len(shard_lengths)
            if previous_version_id is not None: next_version_id_dict[previous_version_id] = version_id
            previous_version_id = version_id
            shard_lengths.extend(dataset_info["splits"][0]["shardLengths"])
            # logging.info(f"total_num_bytes: {total_num_bytes}")
            # logging.info(f"shard_lengths_accumulated[-1]: {shard_lengths_accumulated[-1]}")

            # check num of files in the version folder that starts with "laion400m-train.tfrecord"
            # should be consistent with the number of shardLengths
            folder = "/".join(blob.name.split("/")[:-1]) + "/"
            # logging.info(f"folder: {folder}")
            blobs_in_folder = list(bucket.list_blobs(prefix=folder))
            # logging.info(f"len(blobs_in_folder): {len(blobs_in_folder)}")
            num_files = len([blob for blob in blobs_in_folder if blob.name.split("/")[-1].startswith("laion400m-train.tfrecord")])
            # logging.info(f"num_files: {num_files}")
            if num_files != len(dataset_info["splits"][0]["shardLengths"]):
                logging.info(f"num_files: {num_files} != len(dataset_info['splits'][0]['shardLengths']): {len(dataset_info['splits'][0]['shardLengths'])}")
                exit()
    shard_lengths_accumulated[final_version_id+1] = len(shard_lengths)
    assert previous_version_id == final_version_id
    next_version_id_dict[previous_version_id] = list(shard_lengths_accumulated.keys())[-1]
    logging.info(f"{worker_id}: len(shard_lengths_accumulated): {len(shard_lengths_accumulated)}")
    logging.info(f"{worker_id}: next_version_id_dict: {next_version_id_dict}")
    # logging.info(f"{worker_id}: shard_lengths_accumulated: {shard_lengths_accumulated}")

    # Update metadata in folder 1.0.0
    first_folder_blob = bucket.get_blob(f"{prefix}dataset_info.json")
    if first_folder_blob and worker_id==0:
        dataset_info = json.loads(first_folder_blob.download_as_string())
        logging.info(f"dataset_info['splits'][0]['numBytes']: {dataset_info['splits'][0]['numBytes']}")
        logging.info(f"len(dataset_info['splits'][0]['shardLengths']): {len(dataset_info['splits'][0]['shardLengths'])}")
        logging.info(f"str(total_num_bytes): {str(total_num_bytes)}")
        logging.info(f"len(shard_lengths): {len(shard_lengths)}")
        dataset_info["splits"][0]["numBytes"] = str(total_num_bytes)
        dataset_info["splits"][0]["shardLengths"] = shard_lengths
        # upload the modified metadata
        first_folder_blob.upload_from_string(json.dumps(dataset_info))

    # check the metadata in folder 1.0.0
    first_folder_blob = bucket.get_blob(f"{prefix}dataset_info.json")
    if first_folder_blob and worker_id==0:
        dataset_info = json.loads(first_folder_blob.download_as_string())
        logging.info(f"dataset_info['splits'][0]['numBytes']: {dataset_info['splits'][0]['numBytes']}")
        logging.info(f"len(dataset_info['splits'][0]['shardLengths']): {len(dataset_info['splits'][0]['shardLengths'])}")
        total_num_samples = sum([int(shard_length) for shard_length in dataset_info["splits"][0]["shardLengths"]])
        logging.info(f"total_num_samples: {total_num_samples}")

    total_num_files = shard_lengths_accumulated[final_version_id+1]
    # logging.info(f"total_num_files: {total_num_files}")

    blobs_sorted = [blob for blob in blobs_sorted if blob.name.split("/")[-1].startswith("laion400m-train.tfrecord")]
    # logging.info(f"{worker_id}: len(blobs_sorted): {len(blobs_sorted)}")

    new_file_index_list = []
    file_index_dict = {}
    # copy the files to the destination directory with the modified names
    for i, blob in enumerate(blobs_sorted):
        # logging.info(f"blob.name: {blob.name}")
        version_id = int(blob.name.split("/")[3].split(".")[2].split("_")[0])
        # logging.info(f"version_id: {version_id}")
        if version_id in file_index_dict: 
            file_index_dict[version_id] += 1
            if file_index_dict[version_id] >= shard_lengths_accumulated[next_version_id_dict[version_id]]:
                # check race condition
                logging.info(f"file_index_dict[{version_id}]: {file_index_dict[version_id]} >= shard_lengths_accumulated[{next_version_id_dict[version_id]}]: {shard_lengths_accumulated[next_version_id_dict[version_id]]}")
                exit()
        else: 
            file_index_dict[version_id] = shard_lengths_accumulated[version_id]
        new_file_index = file_index_dict[version_id]
        assert new_file_index < total_num_files
        assert new_file_index < shard_lengths_accumulated[next_version_id_dict[version_id]]
        new_file_index_list.append(new_file_index)
        if new_file_index%num_workers==worker_id:
            logging.info(f"{worker_id}: blob.name: {blob.name}")
            new_file_name = f"laion400m-train.tfrecord-{new_file_index:05d}-of-{total_num_files:05d}"
            new_blob_name = f"{'/'.join(destination_dir.split('/')[3:])}{new_file_name}"
            logging.info(f"{worker_id}: new_blob_name: {new_blob_name}")
            new_blob = bucket.blob(new_blob_name)
            new_blob.upload_from_string(blob.download_as_string())

    # logging.info(f"len(new_file_index_list): {len(new_file_index_list)}")
    assert new_file_index_list[-1] == total_num_files-1
    assert len(new_file_index_list) == total_num_files
    logging.info(f"{worker_id}: len(new_file_index_list): {len(new_file_index_list)}")
    new_file_index_list = sorted(new_file_index_list)
    for i in range(0, len(new_file_index_list)-1):
        if new_file_index_list[i+1] - new_file_index_list[i] != 1:
            logging.info(f"new_file_index_list[i+1] - new_file_index_list[i]: {new_file_index_list[i+1] - new_file_index_list[i]}")
            logging.info(f"new_file_index_list[i]: {new_file_index_list[i]}")
            logging.info(f"new_file_index_list[i+1]: {new_file_index_list[i+1]}")
            exit()
    
    logging.info(f"{worker_id}: Done")
    
    # check the number of files in the destination directory
    blobs = list(bucket.list_blobs(prefix=prefix))
    blobs = [blob for blob in blobs if blob.name.split("/")[3].split(".")[2]=="0"]
    blobs = [blob for blob in blobs if blob.name.split("/")[-1].startswith("laion400m-train.tfrecord")]
    blobs = [blob for blob in blobs if blob.name.split("/")[-1].endswith(f"{total_num_files:05d}")]
    logging.info(f"{worker_id}: len(blobs): {len(blobs)}")

def main(argv):
    logging.info(f"{FLAGS.worker_id}: args.destination_dir: {FLAGS.destination_dir}")
    logging.info(f"{FLAGS.worker_id}: args.final_version_id: {FLAGS.final_version_id}")
    logging.info(f"{FLAGS.worker_id}: args.num_workers: {FLAGS.num_workers}")
    logging.info(f"{FLAGS.worker_id}: args.worker_id: {FLAGS.worker_id}")
    move_and_rename_files(FLAGS.destination_dir, FLAGS.final_version_id, FLAGS.num_workers, FLAGS.worker_id)

if __name__ == "__main__":
    app.run(main)