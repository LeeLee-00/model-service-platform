"""Shared streaming utilities for all model types."""

import asyncio
from typing import Dict, Any, Tuple

from transformers import GenerationConfig

from app.core.config import settings


def make_generation_config(temperature, top_p, max_new_tokens):
    """Convenience factory for one-off `GenerationConfig` instances.

    Parameters
    ----------
    temperature : float | None
        Sampling temperature; if falsy/0, greedy decoding is used.
    top_p : float | None
        Nucleus sampling parameter; only used when sampling.
    max_new_tokens : int
        Maximum number of new tokens to generate.

    Returns
    -------
    transformers.GenerationConfig
        A config suitable for passing to the HF pipeline.
    """
    do_sample = (temperature or 0) > 0
    kwargs = {
        "do_sample": do_sample,
        "max_new_tokens": int(max_new_tokens),
        "num_return_sequences": 1,
        "return_full_text": False,
    }
    if do_sample:
        kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})
    return GenerationConfig(**kwargs)


def normalize_stops(stop: Any) -> list[str]:
    """Normalize OpenAI-style `stop` parameter to a small, unique string list.

    Parameters
    ----------
    stop : Any
        None, a single string, or a list of strings.

    Returns
    -------
    list[str]
        Deduplicated, non-empty stop strings (maximum of 4 per OpenAI spec).
    """

    if stop is None:
        return []
    if isinstance(stop, str):
        stop_list = [stop]
    elif isinstance(stop, list):
        stop_list = [s for s in stop if isinstance(s, str)]
    else:
        return []
    seen, out = set(), []
    for s in stop_list:
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out[:4]  # OpenAI allows up to 4 stops


def group_requests(
    batch: list,
) -> Dict[Tuple[Tuple[float, float, int, int], Tuple[str, ...]], list]:
    """Group batch items by generation signature and stop sequences.

    Items are grouped by a composite key:
        (gen_sig, tuple(stops))
    so that requests with identical sampling/length parameters and identical
    stop lists can be served by a single batched pipeline invocation.

    Parameters
    ----------
    batch : list
        Output of `collect_batch`, a list of 4-tuples.

    Returns
    -------
    dict
        Mapping of group_key → list[(messages, gen_kwargs, response_q)],
        where `group_key` is `(gen_sig, tuple(stops))`.

    Notes
    -----
    - Unknown or missing `stops` are normalized to an empty tuple.
    - The `messages` field is passed through unchanged for prompt building.
    """
    groups: Dict[Tuple[Tuple[float, float, int, int], Tuple[str, ...]], list] = {}
    for item in batch:
        messages, gen_sig, gen_kwargs, response_q = item
        stops_key = tuple(gen_kwargs.get("stops") or [])
        group_key = (gen_sig, stops_key)
        groups.setdefault(group_key, []).append((messages, gen_kwargs, response_q))
    return groups


def create_chunk_dict(stream_id, created, delta_content, finish_reason=None):
    """Create a standard streaming chunk dictionary.

    Args:
        stream_id: Unique identifier for the streaming session
        created: Unix timestamp for the chunk
        delta_content: Dictionary containing the delta content (e.g.,
                        {"role": "assistant", "content": "text"})
        finish_reason: Optional finish reason ("stop", "length", None)

    Returns:
        Dictionary representing a chat completion chunk
    """
    return {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "choices": [
            {
                "delta": delta_content,
                "index": 0,
                "finish_reason": finish_reason,
            }
        ],
        "model": settings.MODEL_NAME,
    }


def create_streaming_response_headers() -> Dict[str, str]:
    """Standard headers for streaming responses."""
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


def create_error_chunk(error_message: str) -> Dict[str, Any]:
    """Create standardized error chunk."""
    return {"error": {"message": error_message, "type": "server_error"}}


async def collect_batch(queue: asyncio.Queue) -> list:
    """Collect a batch of work items from the shared queue.

    Starts with one blocking `queue.get()` to ensure progress, then attempts
    to pull additional items until either `settings.BATCH_SIZE` is reached
    or `settings.BATCH_TIMEOUT` elapses without a new item.

    Parameters
    ----------
    queue : asyncio.Queue
        The inbound work queue shared with the API layer.

    Returns
    -------
    list
        A list of 4-tuples `(messages, gen_sig, gen_kwargs, response_q)`.

    Notes
    -----
    - This function never returns an empty list.
    - Timeout only applies after the first item has been received.
    """
    first = await queue.get()
    batch = [first]
    while len(batch) < settings.BATCH_SIZE:
        try:
            nxt = await asyncio.wait_for(queue.get(), timeout=settings.BATCH_TIMEOUT)
            batch.append(nxt)
        except asyncio.TimeoutError:
            break
    return batch


def add_chunk_usage(chunk: dict, prompt_tokens: int, completion_tokens: int) -> dict:
    """Add usage information to a streaming chunk.

    Parameters
    ----------
    chunk : dict
        The streaming chunk to modify
    prompt_tokens : int
        Number of prompt tokens
    completion_tokens : int
        Number of completion tokens

    Returns
    -------
    dict
        The modified chunk with usage information
    """
    chunk["usage"] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    return chunk
