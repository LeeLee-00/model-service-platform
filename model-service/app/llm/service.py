"""Background model worker: batching, grouping, and inference orchestration.

This module implements the **non-streaming** execution engine that powers
the `/v1/chat/completions` endpoint when `stream=False`. It consumes
work items from a shared `asyncio.Queue`, batches compatible requests,
invokes the Hugging Face generation pipeline in batch mode, and returns
results to per-request response queues.

Work item contract
------------------
Each queue item MUST be a 4-tuple:
    (messages, gen_sig, gen_kwargs, response_q)

- messages : list[ChatMessage] | ChatMessage
    Chat history used to build a single prompt.
- gen_sig : tuple(float, float, int, int)
    (temperature, top_p, max_new_tokens, n) — used for grouping.
- gen_kwargs : dict
    Additional per-request generation kwargs. Expected: {"stops": list[str]}.
- response_q : asyncio.Queue
    The per-request queue where a result dict is posted:
      {"texts": list[str], "finish_reasons": list[str]}
    or an error dict:
      {"error": str}

Generation
----------
For each group:
1) Build prompts from messages (`to_prompt_from_messages`).
2) Prepare a `GenerationConfig` and kwargs (`prepare_generation_with_stops`).
3) Call the pipeline on all prompts in one batched invocation.
4) Post-process (`process_group_outputs`) and deliver results to callers.

This path is separate from the streaming path, which bypasses batching to
minimize latency.
"""

import asyncio
from typing import List

from app.core.config import settings
from app.core.logger import setup_logger
from app.chat.util import collect_batch, group_requests
from app.chat.pipe import get_pipe
from app.llm.util import (
    prepare_generation_with_stops,
    process_group_outputs,
    to_prompt_from_messages,
)

logger = setup_logger("model_service.llm.service")


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
async def model_server_loop(queue: asyncio.Queue):
    """Main background loop that executes batched text generation.

    This coroutine runs indefinitely:
      - Collect a time/size-bounded batch from `queue`.
      - Group items by `(gen_sig, stops)`.
      - For each group, build prompts and a `GenerationConfig`,
        run a single batched pipeline call, post-process outputs,
        and return results to each request's `response_q`.

    Parameters
    ----------
    queue : asyncio.Queue
        Shared inbound work queue from the API layer.

    Returns
    -------
    None
        This function never returns under normal operation.

    Error Handling
    --------------
    If a batched pipeline call raises a recoverable exception
    (e.g., `RuntimeError`, `ValueError`, `OSError`), the error is logged
    and an error dict is pushed to **each** group's `response_q`. The loop
    then continues processing subsequent batches.

    Notes
    -----
    - The Hugging Face pipeline is initialized once via `get_pipe()`
      before entering the processing loop to amortize startup overhead.
    - Token-level stopping is only attached when there is a single prompt
      in the group; text-level trimming is always applied during post-process.
    """
    logger.info("Starting model server loop")
    pipe = await get_pipe()
    logger.info(
        "Model pipeline initialized successfully, running on device: %s",
        settings.DEVICE,
    )
    tokenizer = pipe.tokenizer

    while True:  # pylint: disable=too-many-locals
        batch = await collect_batch(queue)
        logger.info("Processing batch window with %d queued items", len(batch))
        groups = group_requests(batch)

        for group_key, items in groups.items():
            gen_sig, shared_stop_key = group_key
            _, _, max_new_tokens, n = gen_sig
            shared_stops = list(shared_stop_key)

            prompts: List[str] = []
            response_qs: List[asyncio.Queue] = []
            for messages, _, rq in items:
                prompt = to_prompt_from_messages(
                    messages if isinstance(messages, list) else [messages], tokenizer
                )
                prompts.append(prompt)
                response_qs.append(rq)

            gen_config, gen_kwargs = prepare_generation_with_stops(
                gen_sig, shared_stops, tokenizer, len(prompts)
            )

            logger.debug("Batched call: size=%d, sig=%s", len(prompts), gen_sig)
            try:
                logger.debug(
                    "generation_config (batched): %s", gen_config.to_dict()
                )  # pylint: disable=line-too-long

                outputs = pipe(
                    prompts,
                    batch_size=len(prompts),
                    generation_config=gen_config,
                    **gen_kwargs,
                )
            except (RuntimeError, ValueError, OSError) as exc:
                logger.error("Error processing batch: %s", exc)
                for rq in response_qs:
                    await rq.put({"error": str(exc)})
                continue

            texts, finish_reasons = process_group_outputs(
                outputs, n, shared_stops, tokenizer, max_new_tokens
            )

            for rq in response_qs:
                await rq.put({"texts": texts, "finish_reasons": finish_reasons})
