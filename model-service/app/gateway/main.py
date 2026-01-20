"""
Gateway Service

Unified API gateway for routing requests to model services.
Provides intelligent load balancing, health monitoring, and service discovery.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import httpx
import json

from app.core.logger import setup_logger
from app.core.registry import registry, ServiceHealth, GPUStats

logger = setup_logger("gateway")

app = FastAPI(
    title="Model Service Gateway",
    description="Unified gateway for multi-GPU model cluster",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request"""
    input: str | List[str]
    model: Optional[str] = "default"


@app.get("/health")
async def health_check():
    """Gateway health check"""
    return {
        "status": "healthy",
        "service": "gateway",
        "version": "1.0.0"
    }


@app.get("/services")
async def get_services():
    """
    Get status of all backend services
    
    Returns health status, response times, and GPU statistics
    """
    health = await registry.health_check()
    gpu_stats = await registry.get_gpu_stats()
    
    return {
        "services": {
            name: {
                "healthy": status.healthy,
                "url": status.url,
                "response_time_ms": status.response_time_ms,
                "last_check": status.last_check.isoformat(),
                "error": status.error,
                "gpu_stats": [gpu.dict() for gpu in gpu_stats.get(name, [])]
            }
            for name, status in health.items()
        },
        "summary": {
            "total_services": len(health),
            "healthy_services": sum(1 for s in health.values() if s.healthy),
            "unhealthy_services": sum(1 for s in health.values() if not s.healthy)
        }
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    Unified chat completions endpoint with intelligent routing
    
    Routes to best available LLM service based on health and load
    """
    # Find best available LLM service
    service_name = await registry.get_best_service("llm")
    
    if not service_name:
        raise HTTPException(
            status_code=503,
            detail="No LLM services available. Check service health."
        )
    
    service_url = registry.get_service_url(service_name)
    logger.info(f"Routing chat request to {service_name}")
    
    try:
        async with httpx.AsyncClient() as client:
            # Proxy request to backend service
            response = await client.post(
                f"{service_url}/v1/chat/completions",
                json=request.dict(exclude_none=True),
                timeout=60.0
            )
            
            # Handle streaming responses
            if request.stream:
                async def stream_proxy():
                    async for chunk in response.aiter_bytes():
                        yield chunk
                
                return StreamingResponse(
                    stream_proxy(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache"}
                )
            
            # Handle non-streaming responses
            return response.json()
    
    except httpx.TimeoutException:
        logger.error(f"Request to {service_name} timed out")
        raise HTTPException(status_code=504, detail="Service timeout")
    
    except Exception as e:
        logger.error(f"Error proxying to {service_name}: {e}")
        raise HTTPException(status_code=502, detail=f"Backend error: {str(e)}")


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    """
    Unified embeddings endpoint
    
    Routes to embedding service
    """
    service_name = await registry.get_best_service("embedding")
    
    if not service_name:
        raise HTTPException(
            status_code=503,
            detail="Embedding service not available"
        )
    
    service_url = registry.get_service_url(service_name)
    logger.info(f"Routing embedding request to {service_name}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{service_url}/v1/embeddings",
                json=request.dict(),
                timeout=30.0
            )
            return response.json()
    
    except Exception as e:
        logger.error(f"Error proxying to {service_name}: {e}")
        raise HTTPException(status_code=502, detail=f"Backend error: {str(e)}")


@app.get("/v1/models")
async def list_models():
    """
    List all available models across services
    
    Aggregates models from all healthy services
    """
    health = await registry.health_check()
    all_models = []
    
    async with httpx.AsyncClient() as client:
        for name, status in health.items():
            if not status.healthy:
                continue
            
            try:
                response = await client.get(
                    f"{status.url}/v1/models",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "data" in data:
                        # Add service info to each model
                        for model in data["data"]:
                            model["service"] = name
                            all_models.append(model)
            
            except Exception as e:
                logger.debug(f"Failed to get models from {name}: {e}")
    
    return {
        "object": "list",
        "data": all_models
    }


@app.get("/gpu-stats")
async def get_gpu_stats():
    """
    Aggregate GPU statistics from all services
    
    Provides cluster-wide GPU utilization view
    """
    stats = await registry.get_gpu_stats()
    
    # Calculate totals
    total_memory_used = 0
    total_memory_available = 0
    gpu_count = 0
    
    for service_stats in stats.values():
        for gpu in service_stats:
            total_memory_used += gpu.memory_used_mb
            total_memory_available += gpu.memory_total_mb
            gpu_count += 1
    
    return {
        "services": {
            name: [gpu.dict() for gpu in gpus]
            for name, gpus in stats.items()
        },
        "summary": {
            "total_gpus": gpu_count,
            "total_memory_used_mb": round(total_memory_used, 2),
            "total_memory_available_mb": round(total_memory_available, 2),
            "cluster_utilization_percent": round(
                (total_memory_used / total_memory_available * 100) if total_memory_available > 0 else 0,
                2
            )
        }
    }


@app.post("/services/{service_name}/proxy")
async def proxy_to_service(service_name: str, request: Request):
    """
    Generic proxy endpoint for direct service access
    
    Allows bypassing intelligent routing when needed
    """
    service_url = registry.get_service_url(service_name)
    
    if not service_url:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
    
    # Get request body
    body = await request.body()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=request.method,
                url=f"{service_url}{request.url.path}",
                content=body,
                headers=dict(request.headers),
                timeout=60.0
            )
            
            return JSONResponse(
                content=response.json() if response.headers.get("content-type") == "application/json" else response.text,
                status_code=response.status_code
            )
    
    except Exception as e:
        logger.error(f"Error proxying to {service_name}: {e}")
        raise HTTPException(status_code=502, detail=f"Backend error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
