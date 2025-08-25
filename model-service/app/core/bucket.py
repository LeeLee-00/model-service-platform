"""This module provides functions to interact with a MinIO S3 bucket for model storage."""

import os
from pathlib import Path
import shutil
from typing import Optional

from app.core.config import settings
from app.core.logger import setup_logger

import boto3
from botocore.exceptions import ClientError
from botocore.client import Config, BaseClient
from huggingface_hub import hf_hub_download, list_repo_files

logger = setup_logger("model_service.bucket")


def get_s3_client() -> Optional[BaseClient]:
    """
    Initializes and returns an S3 client using the specified endpoint URL.

    :return: An S3 client object if successful, None otherwise.
    """
    try:

        _s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.MINIO_SVC_USER,
            aws_secret_access_key=settings.MINIO_SVC_PASSWORD,
            endpoint_url=settings.MINIO_ENDPOINT,
            config=Config(signature_version="s3v4"),
        )
        return _s3_client
    except ConnectionError as ex:
        logger.error("Error initializing S3 client: %s", ex)
        return None


def model_exists_in_minio() -> bool:
    """
    Checks if the model exists in the MinIO bucket.

    :return: True if the model exists in the bucket, False otherwise.
    """
    s3 = get_s3_client()
    if not s3:
        logger.error("S3 client is not initialized, cannot check model existence.")
        return False
    bucket = settings.MINIO_BUCKET

    try:
        response = s3.list_objects_v2(
            Bucket=bucket, Prefix=settings.MODEL_HF_DIR_NAME + "/", MaxKeys=1
        )
        return "Contents" in response
    except ClientError as e:
        logger.error("Error checking if model exists in MinIO: %s", e)
        return False


def model_exists_locally() -> bool:
    """
    Checks if the model directory exists locally and is not empty.

    :return: True if the model directory exists and is not empty, False otherwise.
    """
    model_path = Path(settings.MODELS_DIR) / settings.MODEL_HF_DIR_NAME
    return model_path.exists() and any(model_path.iterdir())


def download_model_from_minio():
    """
    Downloads the model files from the MinIO bucket to the local models directory.
    """
    s3 = get_s3_client()
    if not s3:
        logger.error("S3 client is not initialized, cannot check model existence.")
        raise RuntimeError("Failed to initialize S3 client")
    # List model objects in minio
    response = s3.list_objects_v2(
        Bucket=settings.MINIO_BUCKET, Prefix=settings.MODEL_HF_DIR_NAME + "/"
    )

    if "Contents" not in response:
        logger.error("Failed to find any objects for %s in bucket", settings.MODEL_NAME)
        raise RuntimeError("Failed to find model objects")
    logger.info(
        "Found %d objects for %s in bucket",
        len(response["Contents"]),
        settings.MODEL_NAME,
    )

    for obj in response["Contents"]:
        key = obj["Key"]
        filename = key.split("/")[-1]
        local_path = os.path.join(
            settings.MODELS_DIR, settings.MODEL_HF_DIR_NAME, filename
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(Bucket=settings.MINIO_BUCKET, Key=key, Filename=local_path)


def download_model_from_hf():
    """
    Downloads the model from Hugging Face and saves it to the local models directory.
    """
    try:
        logger.info("Downloading Hugging Face model '%s'...", settings.MODEL_NAME)
        model_path = Path(settings.MODELS_DIR) / settings.MODEL_HF_DIR_NAME
        model_path.mkdir(parents=True, exist_ok=True)

        files = list_repo_files(settings.MODEL_NAME, token=settings.HF_TOKEN)
        for file in files:
            hf_hub_download(
                repo_id=settings.MODEL_NAME,
                filename=file,
                token=settings.HF_TOKEN,
                cache_dir=None,
                local_dir=model_path,
            )

        logger.info("Downloaded model to %s", model_path)

    except (ValueError, ConnectionError, OSError) as exc:
        logger.error("Failed to download model: %s", exc)
        raise RuntimeError("Failed to download from Huggingface") from exc


def upload_model_to_minio():
    """
    Uploads model files from the specified local path to the MinIO bucket.

    :return: None
    """
    model_path = Path(settings.MODELS_DIR) / settings.MODEL_HF_DIR_NAME
    logger.info("Uploading model files from %s to MinIO", model_path)
    s3 = get_s3_client()
    if not s3:
        logger.error("S3 client is not initialized, cannot check model existence.")
        raise RuntimeError("Failed to initialize S3 client")

    # Remove .cache directory before upload
    cache_dir = model_path / ".cache"
    try:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logger.info("Removed .cache directory from %s", cache_dir)
    except (PermissionError, OSError) as exc:
        logger.warning("Failed to remove .cache directory: %s", exc)

    for filepath in model_path.rglob("*"):
        if filepath.is_file():
            s3.upload_file(
                str(filepath),
                settings.MINIO_BUCKET,
                f"{settings.MODEL_HF_DIR_NAME}/{filepath.relative_to(model_path)}",
            )
