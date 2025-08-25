"""
This module provides the main service loop for the model server, handling text embedding requests.
"""

import asyncio

import numpy as np
from transformers import pipeline

from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger("model_service.embedding.service")


async def model_server_loop(queue: asyncio.Queue):
    """Main model server loop for processing embedding requests."""
    try:
        pipe = pipeline(
            task="feature-extraction",
            model=f"{settings.MODELS_DIR}/{settings.MODEL_HF_DIR_NAME}",
            device=settings.DEVICE,
        )
    except (ValueError, OSError) as exc:
        logger.error("Error initializing pipeline: %s", exc)
        raise

    while True:
        texts, response_q = await queue.get()
        batch_texts = [texts]
        queues = [response_q]

        # Try to collect more requests for batching
        while True:
            try:
                texts, response_q = await asyncio.wait_for(queue.get(), timeout=0.01)
                batch_texts.append(texts)
                queues.append(response_q)
            except asyncio.TimeoutError:
                break

        try:
            outputs = pipe(batch_texts, batch_size=len(batch_texts))
            for q, embedding_array in zip(queues, outputs):
                embedding = np.array(embedding_array).mean(axis=0)
                # Normalize
                embedding = embedding / np.linalg.norm(embedding)
                await q.put({"embedding": embedding.tolist()})
        except (RuntimeError, ValueError, OSError) as exc:
            logger.error("Error during embedding generation: %s", exc)
            for q in queues:
                await q.put({"error": str(exc)})
