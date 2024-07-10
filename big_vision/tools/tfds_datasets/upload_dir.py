import argparse
from pathlib import Path
from google.cloud.storage import Client, transfer_manager

def upload_directory_with_transfer_manager(bucket_name, source_directory, destination_blob, workers=8):
    """Upload every file in a directory, including all files in subdirectories.
    
    Each blob name is derived from the filename, not including the `directory`
    parameter itself. For complete control of the blob name for each file (and
    other aspects of individual blob metadata), use
    transfer_manager.upload_many() instead.
    """


    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)

    # Generate a list of paths (in string form) relative to the `directory`.
    # This can be done in a single list comprehension, but is expanded into
    # multiple lines here for clarity.

    # First, recursively get all files in `directory` as Path objects.
    directory_as_path_obj = Path(source_directory)
    paths = directory_as_path_obj.rglob("*")

    # Filter so the list only includes files, not directories themselves.
    file_paths = [path for path in paths if path.is_file()]

    # These paths are relative to the current working directory. Next, make them
    # relative to `directory`
    relative_paths = [path.relative_to(source_directory) for path in file_paths]

    # Finally, convert them all to strings.
    string_paths = [str(path) for path in relative_paths]

    print("Found {} files.".format(len(string_paths)))

    # Start the upload.
    results = transfer_manager.upload_many_from_filenames(
        bucket, string_paths, source_directory=source_directory, blob_name_prefix=destination_blob, max_workers=workers
    )

    for name, result in zip(string_paths, results):
        # The results list is either `None` or an exception for each filename in
        # the input list, in order.

        if isinstance(result, Exception):
            print("Failed to upload {} due to exception: {}".format(name, result))
        else:
            print("Uploaded {} to {}.".format(name, bucket.name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload files from a directory to Google Cloud Storage")
    parser.add_argument("--bucket_name", help="The name of the GCS bucket")
    parser.add_argument("--source_directory", help="The directory to upload")
    parser.add_argument("--destination_blob", help="The name of the blob to create")
    parser.add_argument("--workers", type=int, default=8, help="The maximum number of workers to use for the upload")

    args = parser.parse_args()

    upload_directory_with_transfer_manager(args.bucket_name, args.source_directory, args.destination_blob, args.workers)
