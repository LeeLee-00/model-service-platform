"""Non-streaming chat completion handler.

This module contains the request handler used by the `/v1/chat/completions`
endpoint when `stream=False`. It delegates inference to the background
model server loop via a shared asyncio queue, waits for the result on a
per-request response queue, and returns an OpenAI-compatible JSON payload.

High-level flow:
1) Enqueue: (messages, gen_sig, gen_kwargs, response_q) → model_queue
2) Await:    result dict from response_q (produced by model_server_loop)
3) Format:   Build OpenAI-style response with usage accounting
"""

import asyncio
import time
from typing import List
from uuid import uuid4
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from app.core.config import settings


async def handle_non_streaming_request(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    req,
    model_queue: asyncio.Queue,
    tokenizer,
    prompt_tokens,
    gen_sig,
    gen_kwargs,
    is_multimodal: bool = False,
):
    """Handle a non-streaming chat completion request.

    This function enqueues the request for batched processing by the
    background model server loop (see `model_server_loop`), waits for
    the generated outputs on a per-request response queue, and returns
    an OpenAI-compatible JSON response (single payload, no SSE).

    Parameters
    ----------
    req :
        A ChatCompletionRequest-like object with a `.messages` attribute
        (list of role/content items). Only `.messages` is used here.
    model_queue : asyncio.Queue
        The shared queue consumed by the model server loop. Items placed on
        this queue must be tuples of the form:
        `(messages, gen_sig, gen_kwargs, response_q)`.
    tokenizer :
        A Hugging Face-compatible tokenizer used *only* to count
        completion tokens for usage metrics. If `None`, a naive word
        count is used as a fallback.
    prompt_tokens : int
        Precomputed token count for the prompt portion (used in usage).
    gen_sig : tuple
        Generation signature tuple: `(temperature, top_p, max_new_tokens, n)`.
        Used by the worker to group/batch compatible requests.
    gen_kwargs : dict
        Additional generation kwargs consumed by the worker. Typically includes:
        `{"stops": List[str]}` and may be extended in the future.

    Returns
    -------
    fastapi.responses.JSONResponse
        A JSON response matching OpenAI's `/v1/chat/completions` schema,
        including `choices[*].finish_reason` and `usage` fields.

    Raises
    ------
    fastapi.HTTPException
        If the worker returns an error dict (e.g., OOM, invalid config),
        this raises `HTTPException(status_code=500)` with a server_error payload.

    Notes
    -----
    - This function does not run generation itself; it delegates to the
      background worker for batching and inference.
    - Usage accounting:
        * `prompt_tokens` is provided by the caller.
        * `completion_tokens` is derived here using the tokenizer.
        * `total_tokens = prompt_tokens + completion_tokens`.
    """
    response_q: asyncio.Queue = asyncio.Queue()
    await model_queue.put((req.messages, gen_sig, gen_kwargs, response_q))
    result = await response_q.get()

    if isinstance(result, dict) and "error" in result:
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": result["error"], "type": "server_error"}},
        )

    texts: List[str] = result["texts"]
    finish_reasons: List[str] = result.get("finish_reasons", ["stop"] * len(texts))
    created = int(time.time())

    if is_multimodal:
        # For multimodal, token counting is more complex due to images
        # Use approximation for now
        completion_tokens = sum(len(text.split()) for text in texts)
    else:
        completion_tokens = sum(
            (
                len(tokenizer.encode(text, add_special_tokens=False))
                if tokenizer
                else len(text.split())
            )
            for text in texts
        )
    total_tokens = prompt_tokens + completion_tokens

    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid4().hex}",
            "object": "chat.completion",
            "created": created,
            "model": settings.MODEL_NAME,
            "choices": [
                {
                    "index": i,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": (
                        finish_reasons[i] if i < len(finish_reasons) else "stop"
                    ),
                }
                for i, text in enumerate(texts)
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
    )
