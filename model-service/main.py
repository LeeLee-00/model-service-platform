"""Main application entry point for the model service."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logger import setup_logger
from app.core.bucket import (
    model_exists_in_minio,
    model_exists_locally,
    download_model_from_minio,
    download_model_from_hf,
    upload_model_to_minio,
)

# pylint: disable=invalid-name
model_server_loop = None
# pylint: disable=invalid-name
router = None

if settings.MODEL_TYPE == "EMBEDDING":
    from app.embedding.service import model_server_loop
    from app.embedding.routes import router
elif settings.MODEL_TYPE == "TRANSCRIPTION":
    from app.transcription.service import model_server_loop
    from app.transcription.routes import router
else:
    from app.chat.routes import router

    if settings.MODEL_TYPE == "LLM":
        from app.llm.service import model_server_loop
    elif settings.MODEL_TYPE == "MULTIMODAL":
        from app.multimodal.service import model_server_loop

logger = setup_logger("model_service.main")


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Handle application startup and shutdown."""
    # Startup logic
    model_downloaded = model_exists_in_minio()

    if not model_downloaded:
        logger.info("Model does not exist in Minio, downloading...")
        try:
            download_model_from_hf()
        except Exception as exc:
            logger.error("Failed to download model from hf: %s", exc)
            raise RuntimeError("Models not loaded correctly") from exc

        logger.info("Pushing model to minio")
        try:
            upload_model_to_minio()
        except (ConnectionError, TimeoutError, OSError) as exc:
            logger.error("Failed to push model to minio: %s", exc)

    elif not model_exists_locally():
        logger.info("Model already exists in Minio, fetching to container")
        try:
            download_model_from_minio()
            logger.info("Models downloaded successfully!")
        except Exception as exc:
            logger.error("Failed to download models from minio: %s", exc)
            raise RuntimeError("Models not loaded correctly") from exc

    else:
        logger.info("Model already exists in Minio and locally, skipping download")

    logger.info("Initiliazing model server for %s model types", settings.MODEL_TYPE)
    application.state.model_queue = asyncio.Queue()

    if model_server_loop is None:
        raise RuntimeError(f"Unsupported MODEL_TYPE: {settings.MODEL_TYPE}")

    model_task = asyncio.create_task(model_server_loop(application.state.model_queue))
    yield  # App runs here

    model_task.cancel()
    try:
        await model_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)
app.router.lifespan_context = lifespan

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# *** THIS IS THE MISSING LINE ***
# Include the model-specific routes (chat, embedding, transcription, etc.)
if router:
    app.include_router(router, prefix="/v1")
    logger.info(f"Included {settings.MODEL_TYPE} router with routes: {[route.path for route in router.routes]}")


# OpenAI-like error envelopes
@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request, exc: HTTPException
):  # pylint: disable=unused-argument
    """Handle HTTP exceptions with OpenAI-like error envelopes."""
    detail = exc.detail
    if isinstance(detail, dict):
        # Unwrap if already {"error": {...}}
        err = detail.get("error")
        content = err if isinstance(err, dict) else detail
    else:
        content = {
            "message": str(exc),
            "type": "invalid_request_error",
        }
    return JSONResponse(status_code=exc.status_code, content={"error": content})


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
):  # pylint: disable=unused-argument
    """Handle unhandled exceptions."""
    logger.exception("Unhandled server error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal server error", "type": "server_error"}},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):  # pylint: disable=unused-argument
    """Handle request validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "Validation error",
                "type": "invalid_request_error",
                "details": exc.errors(),
            }
        },
    )


@app.get("/health")
async def health_check():
    """Health check endpoint for service discovery."""
    return {
        "status": "healthy",
        "service": settings.SERVICE_NAME or settings.MODEL_TYPE,
        "model": settings.MODEL_NAME
    }

@app.get("/v1/models")
async def list_models():
    """
    OpenAI-compatible models endpoint
    
    Returns information about the model this service is serving
    """
    # Get clean model name
    clean_model_name = settings.MODEL_NAME.replace("/", "-").replace(".", "-").lower()
    
    return {
        "object": "list",
        "data": [
            {
                "id": clean_model_name,
                "object": "model",
                "created": 1234567890,  # Placeholder timestamp
                "owned_by": "local",
                "permission": [],
                "root": clean_model_name,
                "parent": None,
                "model_type": settings.MODEL_TYPE,
                "service_name": settings.SERVICE_NAME or "unknown"
            }
        ]
    }

@app.get("/gpu-stats")
async def gpu_stats():
    """
    Get GPU statistics for this service.
    
    Returns GPU memory usage, utilization, and temperature.
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {"gpus": [], "cuda_available": False}
        
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**2)  # MB
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**2)  # MB
            memory_total = props.total_memory / (1024**2)  # MB
            
            gpu_info = {
                "device_id": i,
                "name": props.name,
                "memory_used_mb": round(memory_allocated, 2),
                "memory_total_mb": round(memory_total, 2),
                "memory_percent": round((memory_allocated / memory_total * 100), 2),
                "temperature_c": None,  # Requires pynvml
                "utilization_percent": None  # Requires pynvml
            }
            
            # Try to get additional stats with pynvml
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                gpu_info["temperature_c"] = temp
                gpu_info["utilization_percent"] = utilization.gpu
            except:
                pass  # pynvml not available or error
            
            gpus.append(gpu_info)
        
        return {
            "gpus": gpus,
            "cuda_available": True,
            "device_count": torch.cuda.device_count()
        }
    
    except Exception as e:
        logger.error(f"Failed to get GPU stats: {e}")
        return {"gpus": [], "error": str(e)}

