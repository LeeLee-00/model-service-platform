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
    Checks if the model files exist in the MinIO bucket.

    :return: True if model exists, False otherwise.
    """
    s3 = get_s3_client()
    if not s3:
        logger.error("S3 client is not initialized, cannot check model existence.")
        return False

    # Use clean model name
    clean_model_name = settings.MODEL_NAME.replace("/", "-").replace(".", "-").lower()
    
    try:
        response = s3.list_objects_v2(
            Bucket=settings.MINIO_BUCKET, 
            Prefix=f"models/{clean_model_name}/", 
            MaxKeys=1
        )
        return "Contents" in response
    except ClientError as e:
        logger.error("Error checking MinIO for model: %s", e)
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
    
    # Use the same clean name for downloading
    clean_model_name = settings.MODEL_NAME.replace("/", "-").replace(".", "-").lower()
    
    # List model objects in minio
    response = s3.list_objects_v2(
        Bucket=settings.MINIO_BUCKET, Prefix=f"models/{clean_model_name}/"
    )

    if "Contents" not in response:
        logger.error("Failed to find any objects for %s in bucket", clean_model_name)
        raise RuntimeError("Failed to find model objects")
    
    logger.info(
        "Found %d objects for %s in bucket",
        len(response["Contents"]),
        clean_model_name,
    )

    for obj in response["Contents"]:
        key = obj["Key"]
        # Remove the "models/{clean_model_name}/" prefix
        relative_path = key[len(f"models/{clean_model_name}/"):]
        
        if not relative_path:  # Skip the directory itself
            continue
            
        local_path = os.path.join(
            settings.MODELS_DIR, settings.MODEL_HF_DIR_NAME, relative_path
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        logger.debug(f"Downloading {key} to {local_path}")
        s3.download_file(Bucket=settings.MINIO_BUCKET, Key=key, Filename=local_path)
    
    logger.info(f"Model downloaded from models/{clean_model_name}/")


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

    # Use a clean model name for the S3 path
    # Convert "Qwen/Qwen2.5-0.5B-Instruct" to "qwen-2.5-0.5b-instruct"
    clean_model_name = settings.MODEL_NAME.replace("/", "-").replace(".", "-").lower()
    
    logger.info(f"Uploading model to MinIO under name: {clean_model_name}")

    for filepath in model_path.rglob("*"):
        if filepath.is_file():
            relative_path = filepath.relative_to(model_path)
            s3_key = f"models/{clean_model_name}/{relative_path}"
            
            logger.debug(f"Uploading {filepath} to {s3_key}")
            s3.upload_file(
                str(filepath),
                settings.MINIO_BUCKET,
                s3_key
            )
    
    logger.info(f"Model uploaded successfully to models/{clean_model_name}/")
