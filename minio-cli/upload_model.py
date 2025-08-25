"""CLI tool to upload models from Hugging Face to MinIO."""

from pathlib import Path
import logging
import shutil

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from huggingface_hub import list_repo_files, hf_hub_download
import typer

from settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()


def get_s3_client():
    """
    Initializes and returns an S3 client using the specified endpoint URL.

    :return: An S3 client object if successful, None otherwise.
    """
    logger.info("Create s3 object point to endpoint: %s", settings.MINIO_ENDPOINT)
    try:

        _s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.MINIO_SVC_USER,
            aws_secret_access_key=settings.MINIO_SVC_PASSWORD,
            endpoint_url=settings.MINIO_ENDPOINT,
            config=Config(signature_version="s3v4"),
        )
        return _s3_client
    except ClientError as ex:
        logger.error("Error initializing S3 client: %s", ex)
        return None


def get_model_path(model_name: str) -> str:
    """
    Create standardized model directory path.

    :param model_name: The name of the model.
    :return: A standardized path string for the model.
    """
    return "models--" + model_name.replace("/", "--")


def model_exists_in_minio(model_name: str) -> bool:
    """
    Check if the specified model exists in the MinIO bucket.

    :param model_name: The name of the model to check.
    :return: True if the model exists, False otherwise.
    """
    s3 = get_s3_client()
    if not s3:
        logger.error("S3 client is not initialized, cannot check model existence.")
        return False

    bucket = settings.MINIO_BUCKET
    prefix = get_model_path(model_name) + "/"

    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        return "Contents" in response
    except ClientError as e:
        logger.error("Error checking if model exists in MinIO: %s", e)
        return False


def download_hf_model(model_name: str) -> str:
    """
    Downloads the specified Hugging Face model to the local models directory.

    :param model_name: The name of the Hugging Face model to download.
    :return: The local path where the model was downloaded.
    """
    try:
        logger.info("Downloading Hugging Face model '%s'...", model_name)

        model_path = Path.cwd() / "models" / get_model_path(model_name)
        model_path.mkdir(parents=True, exist_ok=True)

        files = list_repo_files(repo_id=model_name, token=settings.HF_TOKEN)
        for file in files:
            hf_hub_download(
                repo_id=model_name,
                filename=file,
                token=settings.HF_TOKEN,
                cache_dir=None,
                local_dir=model_path,
            )

        logger.info("Downloaded model to %s", model_path)
        return model_path
    except (ValueError, ConnectionError, OSError) as e:
        logger.error("Failed to download model: %s", e)
        raise


def upload_model_to_minio(model_path: str, model_name: str):
    """
    Uploads model files from the specified local path to the MinIO bucket.

    :param model_path: The local path where the model files are stored.
    :param model_name: The name of the model being uploaded.
    """
    logger.info("Uploading model files from %s to MinIO", model_path)
    s3 = get_s3_client()
    if not s3:
        logger.error("S3 client is not initialized, cannot upload model.")
        raise RuntimeError("Failed to initialize S3 client")

    local_path = Path(model_path)

    try:
        # Remove .cache directory before upload
        cache_dir = local_path / ".cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logger.info("Removed .cache directory from %s", cache_dir)

        # Upload files to MinIO
        model_dir_name = get_model_path(model_name)
        for filepath in local_path.rglob("*"):
            if filepath.is_file():
                s3.upload_file(
                    str(filepath),
                    settings.MINIO_BUCKET,
                    f"{model_dir_name}/{filepath.relative_to(local_path)}",
                )

        # Delete entire local model directory after successful upload
        logger.info("Removing local model directory: %s", local_path)
        shutil.rmtree(local_path)

    except (PermissionError, OSError) as exc:
        logger.error("Failed to remove cache directory or local files: %s", exc)
        raise RuntimeError("Failed to cleanup local files") from exc
    except ClientError as exc:
        logger.error("Failed to upload model to MinIO: %s", exc)
        raise RuntimeError("Failed to upload to MinIO") from exc


@app.command()
def addmodel(model_name: str):
    """
    Adds a model to the MinIO bucket if it does not already exist.

    :param model_name: The name of the Hugging Face model to add.
    """
    if model_exists_in_minio(model_name):
        logger.info(
            "Model '%s' already exists in MinIO at path s3://%s/%s/",
            model_name,
            settings.MINIO_BUCKET,
            get_model_path(model_name),
        )
        return

    logger.info(
        "Model '%s' not found in MinIO, downloading from Hugging Face...", model_name
    )
    model_path = download_hf_model(model_name)
    upload_model_to_minio(model_path, model_name)


@app.command()
def head():
    """
    Ensures that the specified S3 bucket exists. Creates the bucket if it does not exist.

    :return: None
    """
    s3 = get_s3_client()
    if not s3:
        logger.error("S3 client is not initialized, cannot test connection.")
        return

    try:
        logger.info("Testing connection to %s", settings.MINIO_BUCKET)
        s3.head_bucket(Bucket=settings.MINIO_BUCKET)
        logger.info("Bucket %s is good!", settings.MINIO_BUCKET)
    except ClientError as e:
        logger.error("Error connecting to bucket %s: %s", settings.MINIO_BUCKET, e)


if __name__ == "__main__":
    app()
