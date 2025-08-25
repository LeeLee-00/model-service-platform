"""Streaming multimodal chat completion handler.

This module implements streaming mode for multimodal chat completions.
Since TextIteratorStreamer doesn't work well with image-text-to-text pipelines,
we generate the complete response and then stream it word-by-word to simulate
real-time generation while maintaining the OpenAI SSE format.

High-level flow:
1) Process multimodal messages (decode images, format content)
2) Generate complete response using multimodal pipeline
3) Stream the response word-by-word with appropriate delays
4) Apply stop sequences and finish reason logic
5) Emit final chunk with usage stats if requested
"""

import asyncio
import json

from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.logger import setup_logger
from app.chat.pipe import get_pipe
from app.llm.util import strip_stop
from app.multimodal.util import prepare_multimodal_messages
from app.chat.util import (
    create_chunk_dict,
    create_streaming_response_headers,
    create_error_chunk,
    add_chunk_usage,
)

logger = setup_logger("model_service.multimodal.streaming")


def estimate_prompt_tokens_multimodal(messages):
    """Estimate token count for multimodal messages.

    This is a rough approximation since proper multimodal token counting
    requires the actual tokenizer and image processing logic.

    Args:
        messages: List of processed multimodal messages

    Returns:
        Estimated token count
    """
    total_tokens = 0

    for message in messages:
        if isinstance(message.get("content"), list):
            for item in message["content"]:
                if item.get("type") == "text":
                    # Rough token estimate: ~4 chars per token
                    text = item.get("text", "")
                    total_tokens += len(text) // 4
                elif item.get("type") == "image":
                    # Rough estimate for image tokens (varies by model)
                    # Most vision models use 256-576 tokens per image
                    total_tokens += 400  # Conservative estimate
        elif isinstance(message.get("content"), str):
            total_tokens += len(message["content"]) // 4

    return max(total_tokens, 1)  # Ensure at least 1 token


async def handle_streaming_request(  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-statements
    req,
    gen_config,
    stops,
    request_id,
    stream_id,
    created,
):
    """Handle a streaming multimodal chat completion request (SSE).

    Since multimodal pipelines don't work well with TextIteratorStreamer,
    we generate the complete response first and then stream it token-by-token
    to simulate real-time generation.

    Parameters
    ----------
    req :
        A MultimodalRequest object with messages containing text and images
    gen_config : dict
        Generation configuration (temperature, top_p, max_new_tokens, etc.)
    stops : List[str]
        Stop sequences to terminate generation early
    request_id : str
        Short request identifier for logging
    stream_id : str
        Unique stream identifier included in each SSE chunk
    created : int
        Unix timestamp for the request creation time

    Returns
    -------
    fastapi.responses.StreamingResponse
        A streaming response that yields SSE chunks simulating real-time generation
    """

    async def event_stream(): # pylint: disable=too-many-locals
        """Yield streaming chunks for the multimodal chat completion response (SSE)."""
        try:
            # Get the multimodal pipeline
            pipe = await get_pipe()

            # Process messages (decode images, format content)
            processed_messages = prepare_multimodal_messages(
                req.messages if isinstance(req.messages, list) else [req.messages]
            )
            logger.debug(
                "[%s] Processed %d multimodal messages",
                request_id,
                len(processed_messages),
            )

            result = pipe(
                text=processed_messages,
                return_full_text=False,
                generation_config=gen_config
            )

            if not result or not isinstance(result, list):
                raise ValueError("Invalid response from multimodal pipeline")

            generated_text = result[0].get("generated_text", "")
            logger.debug(
                "[%s] Generated text length: %d chars", request_id, len(generated_text)
            )

            # Apply stop sequences
            original_length = len(generated_text)
            final_text = strip_stop(generated_text, stops)
            hit_stop = len(final_text) < original_length

            if not final_text.strip():
                # Handle empty generation
                logger.warning("[%s] Empty generation result", request_id)
                first_chunk = create_chunk_dict(
                    stream_id,
                    created,
                    {"role": "assistant", "content": ""},
                )
                yield f"data: {json.dumps(first_chunk)}\n\n"

                # Final chunk
                finish_reason = "stop" if hit_stop else "length"
                done_chunk = create_chunk_dict(
                    stream_id, created, {}, finish_reason=finish_reason
                )
                yield f"data: {json.dumps(done_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Stream the text word by word
            words = final_text.split()
            first_chunk = True

            for i, word in enumerate(words):
                # Add space before word except for the first one
                content = word if i == 0 else f" {word}"

                if first_chunk:
                    chunk = create_chunk_dict(
                        stream_id,
                        created,
                        {"role": "assistant", "content": content},
                    )
                    first_chunk = False
                else:
                    chunk = create_chunk_dict(
                        stream_id,
                        created,
                        {"content": content},
                    )

                yield f"data: {json.dumps(chunk)}\n\n"

                # Small delay to simulate streaming (adjust as needed)
                await asyncio.sleep(0.05)

            # Determine finish reason
            finish_reason = "stop" if hit_stop else "length"

            # Estimate token counts for usage
            prompt_tokens = estimate_prompt_tokens_multimodal(processed_messages)
            completion_tokens = len(final_text.split())  # Rough estimate

            # Final chunk with finish reason
            done_chunk = create_chunk_dict(
                stream_id, created, {}, finish_reason=finish_reason
            )

            # Add usage if requested
            include_usage = bool(
                getattr(req, "stream_options", None)
                and getattr(req.stream_options, "include_usage", False)
            )

            if include_usage:
                done_chunk = add_chunk_usage(
                    done_chunk, prompt_tokens, completion_tokens
                )

            yield f"data: {json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"

            logger.debug("[%s] Multimodal streaming completed successfully", request_id)

        except asyncio.CancelledError:
            logger.info("[%s] Multimodal streaming client disconnected", request_id)
            raise
        except (RuntimeError, ValueError, OSError, TypeError) as exc:
            logger.error("[%s] Error in multimodal streaming: %s", request_id, exc)
            err_chunk = create_error_chunk(str(exc))
            yield f"data: {json.dumps(err_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=create_streaming_response_headers(),
    )
