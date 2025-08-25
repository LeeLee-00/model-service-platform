"""Streaming chat completion handler.

This module implements the streaming mode for the `/v1/chat/completions`
endpoint (OpenAI-compatible) using Server-Sent Events (SSE). It leverages
Hugging Face's `TextIteratorStreamer` to receive tokens as they are generated,
and forwards them to the client as incremental `chat.completion.chunk` objects,
terminating with a final `[DONE]` sentinel.

High-level flow:
1) Initialize a `TextIteratorStreamer` bound to the tokenizer.
2) Spawn a background thread that calls `pipe(prompt, streamer=..., ...)`.
3) In an async generator (`event_stream`), iterate tokens from the streamer,
   apply flush heuristics and stop-sequence checks, and yield SSE chunks.
4) On completion, emit a final empty delta containing `finish_reason` and,
   if requested, `usage` (prompt/completion/total tokens), followed by `[DONE]`.

Notes:
- This path intentionally bypasses the batching worker loop for low latency.
- Stop sequences can be enforced both at generation-time (via `stopping_criteria`)
  and at text-time (string search within the aggregated output).
- Flush policy is controlled by settings: MIN_TOKENS, MAX_CHARS,
  WHITESPACE_CHUNK_CHARS, SENTENCE_END.
"""

import asyncio
import json
import threading

from fastapi.responses import StreamingResponse
from transformers import TextIteratorStreamer

from app.core.config import settings
from app.core.logger import setup_logger
from app.chat.util import (
    create_chunk_dict,
    create_streaming_response_headers,
    create_error_chunk,
    add_chunk_usage,
)

logger = setup_logger("model_service.llm.streaming")  #


async def handle_streaming_request(  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals, too-many-statements
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
):
    """Handle a streaming chat completion request (SSE).

    Starts text generation in a background thread using a Hugging Face pipeline
    and streams incremental tokens to the client via an async generator. The
    output format mirrors OpenAI's `chat.completion.chunk` objects and ends
    with a `[DONE]` line.

    Parameters
    ----------
    req :
        A ChatCompletionRequest-like object used to read `max_tokens` and
        optional `stream_options.include_usage`.
    pipe :
        A Hugging Face text-generation pipeline (already initialized).
    tokenizer :
        The tokenizer associated with `pipe`. Used by the streamer and for
        computing completion token counts.
    prompt : str
        Fully prepared model prompt string to generate from.
    prompt_tokens : int
        Precomputed token count of the prompt for usage accounting.
    gen_config :
        A `transformers.GenerationConfig` instance controlling sampling/length.
    stopping_criteria :
        Optional `transformers.StoppingCriteriaList` used by the pipeline to
        halt generation at token-time (may be `None`).
    stops : list[str]
        Textual stop sequences to detect in the aggregated output during
        streaming; complements token-level stopping.
    request_id : str
        Short request identifier (for logging / tracing).
    stream_id : str
        Unique id for the streaming session (surfaced in response chunks).
    created : int
        Unix timestamp used in response chunks.

    Returns
    -------
    fastapi.responses.StreamingResponse
        An SSE response producing `chat.completion.chunk` events and a final
        `[DONE]` sentinel line.

    Raises
    ------
    None directly. Any generation errors are serialized as an SSE chunk with
    an `"error"` object and followed by `[DONE]`.

    Notes
    -----
    - A background thread is used to avoid blocking the event loop while the
      pipeline generates tokens into the `TextIteratorStreamer`.
    - If `req.stream_options.include_usage` is True, the final chunk includes
      a `usage` object with prompt/completion/total token counts.
    """
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    error_holder: dict[str, Exception | None] = {"exc": None}

    def run_generation():
        """Invoke the HF pipeline in a background thread, writing into streamer."""
        try:
            pipe_any = pipe
            kwargs = {
                "streamer": streamer,
                "generation_config": gen_config,
            }
            if stopping_criteria:  # pylint: disable=broad-exception-caught
                kwargs["stopping_criteria"] = stopping_criteria
            pipe_any(prompt, **kwargs)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error_holder["exc"] = exc  # pylint: disable=broad-exception-caught
            try:
                streamer.end()
            except Exception:
                pass

    threading.Thread(target=run_generation, daemon=True).start()

    include_usage = bool(
        getattr(req, "stream_options", None)
        and getattr(req.stream_options, "include_usage", False)
    )

    async def event_stream():
        """Yield streaming chunks for the chat completion response (SSE)."""
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements,invalid-name
        MIN_TOKENS = settings.MIN_TOKENS
        MAX_CHARS = settings.MAX_CHARS
        WHITESPACE_CHUNK_CHARS = settings.WHITESPACE_CHUNK_CHARS
        SENTENCE_END = settings.SENTENCE_END
        max_new_tokens = req.max_tokens or settings.MAX_LENGTH

        first_delta = True
        agg: list[str] = []
        generated_text = ""
        hit_stop = False

        def should_flush(agg_text: str) -> bool:
            """Decide when to flush a partial buffer to the client."""
            if not agg_text:
                return False
            last_char = agg_text[-1]
            if last_char in SENTENCE_END:
                return True
            if len(agg) >= MIN_TOKENS:
                return True
            if len(agg_text) >= MAX_CHARS:
                return True
            if agg_text.endswith(" ") and len(agg_text) >= WHITESPACE_CHUNK_CHARS:
                return True
            return False

        def find_first_stop_ids(text: str) -> int | None:
            """Return the earliest index of any stop sequence in `text`, if present."""
            if not stops:
                return None
            indices = [text.find(stop) for stop in stops if stop]
            indices = [i for i in indices if i != -1]
            return min(indices) if indices else None

        try:
            for token in streamer:
                if token == "":
                    continue

                # The first delta must include the assistant role per OpenAI's format.
                if first_delta:
                    first_delta = False
                    candidate = generated_text + token
                    logger.debug("[%s] first_delta_raw=%r", request_id, candidate)

                    # If a stop sequence appears before the first visible delta,
                    # emit an empty role-bearing chunk and terminate cleanly.
                    stop_idx = find_first_stop_ids(candidate)
                    if stop_idx is not None:
                        chunk = create_chunk_dict(
                            stream_id, created, {"role": "assistant", "content": ""}
                        )
                        yield f"data: {json.dumps(chunk)}\n\n"
                        hit_stop = True
                        break

                    generated_text = candidate
                    chunk = create_chunk_dict(
                        stream_id, created, {"role": "assistant", "content": token}
                    )
                    yield f"data: {json.dumps(chunk)}\n\n"
                    continue

                # Aggregate tokens and decide when to flush based on heuristics.
                agg.append(token)
                agg_text = "".join(agg)
                preview_total = generated_text + agg_text

                # Text-level stop detection (complements token-level criteria).
                stop_idx = find_first_stop_ids(preview_total)
                if stop_idx is not None:
                    to_send = preview_total[:stop_idx]
                    pending = to_send[len(generated_text) :]
                    if pending.strip():
                        chunk = create_chunk_dict(
                            stream_id,
                            created,
                            {"role": "assistant", "content": pending},
                        )
                        yield f"data: {json.dumps(chunk)}\n\n"
                    generated_text = to_send
                    hit_stop = True
                    break

                if should_flush(agg_text):
                    if agg_text.strip():
                        chunk = create_chunk_dict(
                            stream_id,
                            created,
                            {"role": "assistant", "content": agg_text},
                        )
                        yield f"data: {json.dumps(chunk)}\n\n"
                        generated_text += agg_text
                    agg.clear()

            # Flush any tail content if we ended without a stop.
            if not hit_stop and agg:
                tail = "".join(agg)
                if tail.strip():
                    chunk = create_chunk_dict(
                        stream_id, created, {"role": "assistant", "content": tail}
                    )
                    yield f"data: {json.dumps(chunk)}\n\n"
                generated_text += tail

        except asyncio.CancelledError:
            logger.info("Streaming client disconnected")
            raise
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Surface unexpected errors to the client before terminating.
            err_chunk = create_error_chunk(str(exc))
            yield f"data: {json.dumps(err_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # If the generation thread raised, surface that as well.
        if error_holder["exc"] is not None:
            err_chunk = create_error_chunk(str(error_holder["exc"]))
            yield f"data: {json.dumps(err_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Usage accounting for the final chunk (optional).
        if tokenizer is not None:
            completion_tokens = len(
                tokenizer.encode(generated_text, add_special_tokens=False)
            )
        else:
            completion_tokens = 0

        finish_reason = "stop"
        if not hit_stop and completion_tokens >= int(max_new_tokens):
            finish_reason = "length"

        done_chunk = create_chunk_dict(
            stream_id,
            created,
            {"role": "assistant", "content": ""},
            finish_reason=finish_reason,
        )

        if include_usage:
            done_chunk = add_chunk_usage(done_chunk, prompt_tokens, completion_tokens)

        # Final empty delta + optional usage, followed by the DONE sentinel.
        yield f"data: {json.dumps(done_chunk)}\n\ndata: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=create_streaming_response_headers(),
    )
