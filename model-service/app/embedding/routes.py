"""
This module defines the FastAPI routes for handling embedding requests.
"""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logger import setup_logger
from app.embedding.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    EmbeddingUsage,
)

logger = setup_logger("model_service.embedding.routes")

router = APIRouter()


def get_model_queue(request: Request) -> asyncio.Queue:
    """Dependency to get the model queue from the application state."""
    return request.app.state.model_queue


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(
    request: EmbeddingRequest,
    model_queue: Annotated[asyncio.Queue, Depends(get_model_queue)],
):
    """Handle embedding requests."""
    logger.info(
        "Received embedding request for model: %s, input type: %s",
        request.model,
        type(request.input).__name__,
    )
    if isinstance(request.input, str):
        input_texts = [request.input]
    else:
        input_texts = request.input

    if not input_texts:
        return JSONResponse(
            {
                "error": {
                    "message": "Input cannot be empty",
                    "type": "invalid_request_error",
                }
            },
            status_code=400,
        )

    # Create a response queue for each text
    response_queues = []
    for text in input_texts:
        response_q = asyncio.Queue()
        await model_queue.put((text, response_q))  # Single text per queue item
        response_queues.append(response_q)

    # Collect all results
    embeddings_data = []
    for i, response_q in enumerate(response_queues):
        result = await response_q.get()
        if "error" in result:
            return JSONResponse(
                {"error": {"message": result["error"], "type": "server_error"}},
                status_code=500,
            )

        embeddings_data.append(EmbeddingData(embedding=result["embedding"], index=i))
    # Calculate token usage (approximate)
    total_tokens = sum(len(text.split()) for text in input_texts)

    response = EmbeddingResponse(
        data=embeddings_data,
        model=settings.MODEL_NAME,
        usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )
    return response
