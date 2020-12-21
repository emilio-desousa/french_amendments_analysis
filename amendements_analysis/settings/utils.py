import sys
import os
import amendements_analysis.settings.base as stg


def update_progress(progress):
    barLength = 10
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [{0}] {1}% {2}".format(
        "#" * block + "-" * (barLength - block), progress * 100, status
    )
    sys.stdout.write(text)
    sys.stdout.flush()


from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    stg.DATA_DIR, "credentials.json"
)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))


download_blob(
    "camembert_aux_amandes_stockage",
    "lemmatized.csv",
    os.path.join(stg.DATA_DIR, "lemmatized.csv"),
)
