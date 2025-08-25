"""Multimodal service for handling batched multimodal requests."""

import asyncio
from typing import List

from app.core.config import settings
from app.core.logger import setup_logger
from app.chat.pipe import get_pipe
from app.chat.util import collect_batch, group_requests
from app.llm.util import strip_stop
from app.multimodal.util import prepare_multimodal_messages

logger = setup_logger("model_service.multimodal.service")


async def _process_outputs(outputs, response_qs, n, shared_stops):
    """Extract text from pipeline outputs, apply stop sequences, and send to queues.

    Parameters
    ----------
    outputs : list
        Raw pipeline output objects containing generated text
    response_qs : list[asyncio.Queue]
        Queue objects to send results back to requesters
    n : int
        Number of completions to generate per request
    shared_stops : list[str]
        Stop sequences to trim from generated text
    """
    for output, rq in zip(outputs, response_qs):
        try:
            if isinstance(output, list) and output:
                # Extract generated text from pipeline output
                generated_text = output[0].get("generated_text", "")

                # Apply stop trimming if needed
                final_text = strip_stop(generated_text, shared_stops)

                # Determine finish reason
                finish_reason = "stop" if final_text != generated_text else "length"

                # Prepare result for multiple completions (n parameter)
                result = {
                    "texts": [final_text] * n,
                    "finish_reasons": [finish_reason] * n,
                }
            else:
                logger.warning("Invalid output format from multimodal pipeline")
                result = {
                    "texts": [""] * n,
                    "finish_reasons": ["error"] * n,
                }

            await rq.put(result)

        except (ValueError, TypeError, KeyError) as exc:
            logger.error("Error processing multimodal output: %s", exc)
            await rq.put({"error": f"Output processing error: {str(exc)}"})


async def _run_inference_batch(
    pipe,
    message_batches,
    response_qs,
    temperature,
    top_p,
    max_new_tokens,
    n,
    shared_stops,
):  # pylint: disable=too-many-arguments, too-many-positional-arguments
    """Execute multimodal inference on batched messages with given parameters.

    Parameters
    ----------
    pipe : Pipeline
        Multimodal pipeline for text generation
    message_batches : list[list]
        List of processed message sequences for inference
    response_qs : list[asyncio.Queue]
        Queue objects to send results back to requesters
    temperature : float
        Sampling temperature (0 = deterministic)
    top_p : float
        Nucleus sampling parameter
    max_new_tokens : int
        Maximum tokens to generate
    n : int
        Number of completions per request
    shared_stops : list[str]
        Stop sequences shared across the batch

    Error Handling
    --------------
    If pipeline inference raises a recoverable exception
    (e.g., RuntimeError, ValueError, OSError), the error is logged
    and an error dict is pushed to each request's response_q.
    """
    logger.debug(
        "Multimodal batched call: size=%d, temp=%.2f, top_p=%.2f, max_tokens=%d",
        len(message_batches),
        temperature,
        top_p,
        max_new_tokens,
    )

    try:
        # Run inference for each message batch
        outputs = []
        for messages in message_batches:
            result = pipe(
                text=messages,
                max_new_tokens=max_new_tokens,
                return_full_text=False,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=top_p if temperature > 0 else None,
            )
            outputs.append(result)

    except (RuntimeError, ValueError, OSError) as exc:
        logger.error("Error processing multimodal batch: %s", exc)
        for rq in response_qs:
            await rq.put({"error": str(exc)})
        return

    # Process outputs and apply stop sequences
    await _process_outputs(outputs, response_qs, n, shared_stops)


async def _process_message_group(pipe, group_key, items): #pylint: disable=too-many-locals
    """Convert raw messages to pipeline format and run batched inference.

    Parameters
    ----------
    pipe : Pipeline
        Multimodal pipeline for text generation
    group_key : tuple
        Tuple of (generation_params, stop_sequences) for this group
    items : list[tuple]
        List of (messages, _, response_queue) tuples to process

    Notes
    -----
    Messages are processed through prepare_multimodal_messages before
    being passed to the inference batch. Invalid messages result in
    error responses sent to their respective queues.
    """
    gen_sig, shared_stop_key = group_key
    temperature, top_p, max_new_tokens, n = gen_sig
    shared_stops = list(shared_stop_key)

    message_batches: List[List] = []
    response_qs: List[asyncio.Queue] = []

    # Process each request in the group
    for messages, _, rq in items:
        try:
            processed_messages = prepare_multimodal_messages(
                messages if isinstance(messages, list) else [messages]
            )
            message_batches.append(processed_messages)
            response_qs.append(rq)
        except (ValueError, TypeError, KeyError) as exc:
            logger.error("Error processing multimodal messages: %s", exc)
            await rq.put({"error": f"Message processing error: {str(exc)}"})
            continue

    if not message_batches:
        logger.warning("No valid message batches to process")
        return

    await _run_inference_batch(
        pipe,
        message_batches,
        response_qs,
        temperature,
        top_p,
        max_new_tokens,
        n,
        shared_stops,
    )


async def model_server_loop(queue: asyncio.Queue):
    """Main background loop that executes batched multimodal text generation."""
    logger.info("Starting multimodal model server loop")
    pipe = await get_pipe()
    logger.info(
        "Multimodal pipeline initialized successfully, running on device: %s",
        settings.DEVICE,
    )

    while True:
        try:
            batch = await collect_batch(queue)
            logger.info("Processing multimodal batch with %d queued items", len(batch))
            groups = group_requests(batch)

            for group_key, items in groups.items():
                await _process_message_group(pipe, group_key, items)

        except asyncio.CancelledError:
            logger.info("Multimodal model server loop cancelled")
            break
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.error("Unexpected error in multimodal model server loop: %s", exc)
            # Continue the loop to handle other requests
            await asyncio.sleep(1)
