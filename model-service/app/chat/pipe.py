"""Shared pipeline initialization and lifecycle management."""

from __future__ import annotations

import asyncio
from transformers import pipeline

from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger("model_service.core.pipeline")

_PIPE = None
_LOCK = asyncio.Lock()


def _configure_llm_pipeline(pipe):
    """Configure LLM-specific pipeline settings."""
    # Avoid warnings when running greedy decoding by clearing sampling defaults
    try:
        gc = pipe.model.generation_config
        for attr in ("temperature", "top_p", "top_k", "typical_p", "do_sample"):
            if hasattr(gc, attr):
                setattr(gc, attr, None)
        logger.info("Cleared generation_config sampling fields")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("Could not adjust generation_config: %s", exc)

    # Also purge any cached sampling keys from pipeline internals
    try:
        fp = getattr(pipe, "_forward_params", {})
        for k in ("temperature", "top_p", "top_k", "typical_p", "do_sample"):
            if k in fp:
                fp.pop(k, None)
        pipe._forward_params = fp  # pylint: disable=protected-access
        logger.info("Purged sampling keys from pipeline _forward_params")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("Could not adjust pipeline _forward_params: %s", exc)


def _build_pipe():
    """Create and configure the Hugging Face pipeline based on MODEL_TYPE."""
    model_path = f"{settings.MODELS_DIR}/{settings.MODEL_HF_DIR_NAME}"

    # Determine task based on MODEL_TYPE
    if settings.MODEL_TYPE == "LLM":
        task = "text-generation"
    elif settings.MODEL_TYPE == "MULTIMODAL":
        task = "image-text-to-text"
    else:
        raise ValueError(f"Unsupported MODEL_TYPE for pipeline: {settings.MODEL_TYPE}")

    logger.info("Initializing %s pipeline from: %s", task, model_path)

    pipe = pipeline(
        task=task,
        model=model_path,
        device=settings.DEVICE,
    )

    if settings.MODEL_TYPE == "LLM":
        _configure_llm_pipeline(pipe)

    return pipe


async def get_pipe():
    """Return the singleton pipeline, creating it if necessary."""
    global _PIPE  # pylint: disable=global-statement
    if _PIPE is not None:
        return _PIPE
    async with _LOCK:
        if _PIPE is None:
            _PIPE = await asyncio.to_thread(_build_pipe)
            logger.info(
                "%s pipeline ready on device: %s", settings.MODEL_TYPE, settings.DEVICE
            )
        return _PIPE


async def reset_pipe():
    """Reset the cached pipeline instance."""
    global _PIPE  # pylint: disable=global-statement
    async with _LOCK:
        _PIPE = None
