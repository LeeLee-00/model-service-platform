"""Shared routes for all model types."""

import asyncio
import time
from typing import Annotated, Union
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import settings
from app.chat.util import make_generation_config, normalize_stops
from app.chat.pipe import get_pipe
from app.core.logger import setup_logger
from app.llm.handlers.non_streaming import handle_non_streaming_request
from app.llm.util import (
    get_model_queue,
    build_stopping_criteria,
    build_prompt_and_count_tokens,
)
from app.llm.models import ChatCompletionRequest
from app.multimodal.models import MultimodalRequest

if settings.MODEL_TYPE == "MULTIMODAL":
    from app.multimodal.handlers.streaming import handle_streaming_request
else:
    from app.llm.handlers.streaming import handle_streaming_request

logger = setup_logger("model_service.routes")
router = APIRouter()


@router.post("/chat/completions")
async def chat_completions_v1(  # pylint: disable=too-many-locals
    req: Union[ChatCompletionRequest, MultimodalRequest],
    model_queue: Annotated[asyncio.Queue, Depends(get_model_queue)],
):
    """Universal chat completions endpoint for all model types."""

    request_id = uuid4().hex[:8]
    stream_id = f"chatcmpl-{uuid4().hex}"
    created = int(time.time())

    max_new_tokens = req.max_tokens or settings.MAX_LENGTH
    temperature = (
        req.temperature if req.temperature is not None else settings.TEMPERATURE
    )
    top_p = req.top_p if req.top_p is not None else 1.0
    n = req.n or 1

    gen_config = make_generation_config(temperature, top_p, max_new_tokens)
    stops = normalize_stops(req.stop)

    # Streaming validation
    if req.stream and n != 1:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "stream=true only supports n=1",
                    "type": "invalid_request_error",
                    "param": "n",
                }
            },
        )

    if req.stream:
        if settings.MODEL_TYPE == "MULTIMODAL":
            return await handle_streaming_request(
                req, gen_config, stops, request_id, stream_id, created
            )

        pipe = await get_pipe()
        tokenizer = pipe.tokenizer
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        prompt, prompt_tokens = build_prompt_and_count_tokens(msgs, tokenizer)
        stopping_criteria = build_stopping_criteria(stops, tokenizer)

        return await handle_streaming_request(
            req,
            pipe,
            tokenizer,
            prompt,
            prompt_tokens,
            gen_config,
            stopping_criteria,
            stops,
            request_id,
            stream_id,
            created,
        )

    gen_sig = (temperature, top_p, max_new_tokens, n)
    gen_kwargs = {"stops": stops}

    if settings.MODEL_TYPE == "MULTIMODAL":
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        prompt_tokens = sum(
            len(str(msg.get("content", "")).split())
            for msg in msgs
            if isinstance(msg, dict)
        )
        return await handle_non_streaming_request(
            req,
            model_queue,
            None,
            prompt_tokens,
            gen_sig,
            gen_kwargs,
            is_multimodal=True,
        )
    pipe = await get_pipe()
    tokenizer = pipe.tokenizer
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt, prompt_tokens = build_prompt_and_count_tokens(msgs, tokenizer)
    return await handle_non_streaming_request(
        req, model_queue, tokenizer, prompt_tokens, gen_sig, gen_kwargs
    )
