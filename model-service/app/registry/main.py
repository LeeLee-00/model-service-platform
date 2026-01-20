"""
Model Registry Service

Manages models stored in MinIO for offline/air-gapped deployments.
Provides endpoints for listing, downloading, and deleting models.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile
from pathlib import Path

from app.core.logger import setup_logger
from app.core.config import settings
from app.core.bucket import get_s3_client

logger = setup_logger("model_registry")


app = FastAPI(
    title="Model Registry API",
    description="Centralized model management for offline AI deployments",
    version="1.0.0"
)


class ModelDownloadRequest(BaseModel):
    hf_repo: str
    revision: Optional[str] = "main"


class ModelInfo(BaseModel):
    name: str
    size_bytes: Optional[int] = None
    files: List[str] = []


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "model-registry"}


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """
    List all models available in MinIO storage
    
    Returns a list of model names and their metadata
    """
    s3 = get_s3_client()
    if not s3:
        raise HTTPException(status_code=500, detail="S3 client not initialized")
    
    try:
        # List objects in the models bucket with prefix
        response = s3.list_objects_v2(
            Bucket=settings.MINIO_BUCKET,
            Prefix="models/"
        )
        
        if 'Contents' not in response:
            return []
        
        # Group by model name (first directory after models/)
        models_dict = {}
        for obj in response['Contents']:
            key = obj['Key']
            # Parse: models/qwen/config.json -> qwen
            parts = key.split('/')
            if len(parts) >= 3:
                model_name = parts[1]
                if model_name not in models_dict:
                    models_dict[model_name] = {
                        "name": model_name,
                        "size_bytes": 0,
                        "files": []
                    }
                models_dict[model_name]["size_bytes"] += obj['Size']
                models_dict[model_name]["files"].append('/'.join(parts[2:]))
        
        return [ModelInfo(**v) for v in models_dict.values()]
    
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.post("/models/{model_name}/download")
async def download_model(
    model_name: str, 
    request: ModelDownloadRequest,
    background_tasks: BackgroundTasks
):
    """
    Download a model from Hugging Face Hub to MinIO
    
    This enables offline operation by pre-fetching models when internet is available.
    
    Args:
        model_name: Local name to store the model as
        request: HuggingFace repo and revision to download
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="huggingface_hub not installed. Run: pip install huggingface_hub"
        )
    
    def _download_and_upload():
        """Background task to download and upload model"""
        s3 = get_s3_client()
        if not s3:
            logger.error("S3 client not available")
            return
        
        try:
            logger.info(f"Downloading {request.hf_repo} from Hugging Face...")
            
            # Download to temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = snapshot_download(
                    repo_id=request.hf_repo,
                    revision=request.revision,
                    cache_dir=tmpdir,
                    token=os.getenv("HF_TOKEN")
                )
                
                logger.info(f"Uploading {model_name} to MinIO...")
                
                # Upload all files to MinIO using boto3
                model_path = Path(local_path)
                for file_path in model_path.rglob("*"):
                    if file_path.is_file():
                        # Get relative path for object name
                        relative_path = file_path.relative_to(model_path)
                        object_key = f"models/{model_name}/{relative_path}"
                        
                        # Upload to MinIO using S3 API
                        s3.upload_file(
                            str(file_path),
                            settings.MINIO_BUCKET,
                            object_key
                        )
                        logger.debug(f"Uploaded {object_key}")
                
            logger.info(f"Successfully uploaded {model_name} to MinIO")
        
        except Exception as e:
            logger.error(f"Failed to download/upload model: {e}")
            raise
    
    # Run in background
    background_tasks.add_task(_download_and_upload)
    
    return {
        "status": "downloading",
        "model": model_name,
        "hf_repo": request.hf_repo,
        "message": "Model download started in background. Check logs for progress."
    }


@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """
    Delete a model from MinIO storage
    
    Args:
        model_name: Name of the model to delete
    """
    s3 = get_s3_client()
    if not s3:
        raise HTTPException(status_code=500, detail="S3 client not initialized")
    
    try:
        # List all objects for this model
        response = s3.list_objects_v2(
            Bucket=settings.MINIO_BUCKET,
            Prefix=f"models/{model_name}/"
        )
        
        if 'Contents' not in response:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        # Delete each object
        deleted_count = 0
        for obj in response['Contents']:
            s3.delete_object(
                Bucket=settings.MINIO_BUCKET,
                Key=obj['Key']
            )
            deleted_count += 1
        
        logger.info(f"Deleted {deleted_count} files for model {model_name}")
        
        return {
            "status": "deleted",
            "model": model_name,
            "files_deleted": deleted_count
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@app.get("/models/{model_name}")
async def get_model_info(model_name: str) -> ModelInfo:
    """
    Get detailed information about a specific model
    
    Args:
        model_name: Name of the model to query
    """
    s3 = get_s3_client()
    if not s3:
        raise HTTPException(status_code=500, detail="S3 client not initialized")
    
    try:
        response = s3.list_objects_v2(
            Bucket=settings.MINIO_BUCKET,
            Prefix=f"models/{model_name}/"
        )
        
        if 'Contents' not in response:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        total_size = 0
        files = []
        
        for obj in response['Contents']:
            total_size += obj['Size']
            # Remove "models/{model_name}/" prefix
            file_name = obj['Key'][len(f"models/{model_name}/"):]
            files.append(file_name)
        
        return ModelInfo(
            name=model_name,
            size_bytes=total_size,
            files=sorted(files)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
