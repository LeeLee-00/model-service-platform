"""Utilities shared by API handlers and the model server loop.

This module centralizes helper routines for:
- Prompt construction (chat messages → model-ready prompt)
- Stop-sequence handling (token-time and text-time)
- Generation configuration assembly (sampling vs greedy)
- Output post-processing (HF pipeline outputs → final texts + finish reasons)
- App plumbing helpers (e.g., retrieving the shared model queue)

Why centralize?
---------------
Keeping these utilities in one place ensures consistent behavior across
streaming and non-streaming paths and across the API layer and the
background batching worker.

Conventions
-----------
- Token-time stops: implemented via `StoppingCriteriaList(StopOnTokens)`.
- Text-time stops: conservative, case-insensitive string search; used as a
  safeguard to trim any residual text beyond requested stop strings.
- Token counting:
  * Prefer tokenizer-based counting when available.
  * Fall back to naive heuristics only where explicitly noted.

Thread/async safety
-------------------
All functions here are stateless and pure (except `get_model_queue`, which
reads from FastAPI's application state). They are safe to call from any path.
"""

import asyncio
import re
from typing import Any, Dict, List, Tuple

from fastapi import Request
from transformers import GenerationConfig, StoppingCriteriaList

from app.core.logger import setup_logger
from app.llm.models import ChatMessage, StopOnTokens

logger = setup_logger("model_service.llm.util")

# ----------------------------
# Dependencies & Utilities
# ----------------------------


def strip_stop(text: str, stops: list[str]) -> str:
    """
    Trim at the first occurrence of any stop string (case-insensitive).
    This is a post-trim safeguard for cases where token-level stopping wasn't applied.
    """
    if not stops:
        return text
    escaped_stops = [re.escape(s) for s in stops if s]
    if not escaped_stops:
        return text

    pattern = re.compile("|".join(escaped_stops), flags=re.IGNORECASE)
    match = pattern.search(text)
    if match:
        cut = match.start()
        logger.debug("Trimming at stop idx=%s for stops=%s", cut, stops)
        return text[:cut]
    return text


def to_prompt_from_messages(messages: List[ChatMessage], tokenizer) -> str:
    """Convert structured chat messages into a model-ready prompt string.

    Preference is given to chat-template aware tokenizers via
    `tokenizer.apply_chat_template`. If not available or if it fails,
    a simple role-prefixed plain-text format is used as a fallback.

    Parameters
    ----------
    messages : list[ChatMessage]
        Chat items with `.role` and `.content`.
    tokenizer :
        HF tokenizer; if it exposes `apply_chat_template`, that path is used.

    Returns
    -------
    str
        The prompt string to feed into the generation pipeline.
    """
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            msgs = [{"role": m.role, "content": m.content} for m in messages]
            result = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            if isinstance(result, str):
                return result
            logger.warning(
                "apply_chat_template returned non-string type: %s", type(result)
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("apply_chat_template failed, falling back. %s", exc)

    # Fallback plain text format
    lines = [f"{m.role}: {m.content}" for m in messages]
    lines.append("assistant:")
    return "\n".join(lines)


def get_model_queue(request: Request) -> asyncio.Queue:
    """Retrieve the shared model queue from FastAPI application state.

    Parameters
    ----------
    request : fastapi.Request
        The inbound request carrying app state.

    Returns
    -------
    asyncio.Queue
        The queue used to submit jobs to the background model worker.
    """
    return request.app.state.model_queue


def build_stopping_criteria(stops: list[str], tokenizer):
    """Build a token-time stopping criteria list from textual stop strings.

    Parameters
    ----------
    stops : list[str]
        Stop strings to encode into token ids.
    tokenizer :
        HF tokenizer used to encode the stops.

    Returns
    -------
    transformers.StoppingCriteriaList | None
        A stopping criteria list if both stops and tokenizer are provided,
        otherwise None.
    """
    if stops and tokenizer:
        stop_token_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stops]
        return StoppingCriteriaList([StopOnTokens(stop_token_ids)])
    return None


def build_prompt_and_count_tokens(messages, tokenizer):
    """Construct the model prompt and compute its token length.

    Parameters
    ----------
    messages : list[dict]
        Chat messages in role/content dict format (for API side).
    tokenizer :
        HF tokenizer; if it supports `apply_chat_template`, that is used to
        build the prompt and to count tokens via `return_tensors="pt"`.

    Returns
    -------
    tuple[str, int]
        (prompt_str, prompt_token_count)
    """
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = int(prompt_ids.shape[-1])
    else:
        prompt_str = (
            "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        )
        prompt_tokens = (
            len(tokenizer(prompt_str, add_special_tokens=True).input_ids)
            if tokenizer
            else 0
        )
    return prompt_str, prompt_tokens


def prepare_generation_with_stops(
    gen_sig: Tuple[float, float, int, int],
    shared_stops: list[str],
    tokenizer,
    prompt_count: int,
) -> Tuple[GenerationConfig, Dict[str, Any]]:
    """Assemble a `GenerationConfig` and kwargs, optionally with token-time stops.

    Applies token-time stopping only when there is a single prompt in the batch,
    to avoid coupling multi-prompt runs to a shared stop list.

    Parameters
    ----------
    gen_sig : tuple(float, float, int, int)
        (temperature, top_p, max_new_tokens, n)
    shared_stops : list[str]
        Shared textual stops for this group (may be empty).
    tokenizer :
        HF tokenizer used to encode stops into token ids.
    prompt_count : int
        Number of prompts being generated together.

    Returns
    -------
    (GenerationConfig, dict)
        The generation config and supplemental kwargs (e.g., stopping_criteria).
    """
    temperature, top_p, max_new_tokens, n = gen_sig
    stopping_list = None
    if shared_stops and tokenizer is not None and prompt_count == 1:
        stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in shared_stops]
        stopping_list = StoppingCriteriaList([StopOnTokens(stop_ids)])

    gen_kwargs = {
        "num_return_sequences": n,
        "return_full_text": False,
    }
    if stopping_list is not None:
        gen_kwargs["stopping_criteria"] = stopping_list

    if (temperature or 0) > 0:
        gen_config = GenerationConfig(
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
        )
    else:
        gen_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=int(max_new_tokens),
        )

    return gen_config, gen_kwargs


# pylint: disable=too-many-nested-blocks,too-many-locals
def process_group_outputs(
    outputs: Any, n: int, shared_stops: list[str], tokenizer, max_new_tokens: int
) -> Tuple[List[str], List[str]]:
    """Normalize HF pipeline outputs and infer finish reasons.

    Supports both common output shapes:
    - list of dicts or strings (single-prompt with n completions)
    - list of lists of dicts/strings (multi-prompt with n completions each)

    Applies text-time stop trimming and sets `finish_reason` to "length" when
    completion token length meets or exceeds `max_new_tokens`.

    Parameters
    ----------
    outputs : Any
        Raw output from the HF pipeline call.
    n : int
        Number of return sequences per prompt.
    shared_stops : list[str]
        Stop sequences for text-time trimming.
    tokenizer :
        HF tokenizer used for completion token counting.
    max_new_tokens : int
        Generation limit used to infer "length" finish reason.

    Returns
    -------
    (list[str], list[str])
        Finalized texts and their corresponding finish reasons.
    """
    is_list_of_lists = (
        isinstance(outputs, list) and outputs and isinstance(outputs[0], list)
    )

    texts: List[str] = []
    finish_reasons: List[str] = []

    for idx in range(len(outputs) if is_list_of_lists else len(outputs) // n):
        group_out = (
            outputs[idx] if is_list_of_lists else outputs[idx * n : (idx + 1) * n]
        )

        raw_texts: List[str] = []
        for o in group_out:
            if isinstance(o, dict):
                raw_texts.append(o.get("generated_text", ""))
            elif isinstance(o, str):
                raw_texts.append(o)
            else:
                raw_texts.append(str(o))

        for raw in raw_texts:
            txt = strip_stop(raw, shared_stops)
            texts.append(txt)
            reason = "stop"
            if txt == raw:  # not trimmed by stop
                try:
                    if tokenizer is not None:
                        comp_token_len = len(
                            tokenizer.encode(txt, add_special_tokens=False)
                        )
                        if comp_token_len >= int(max_new_tokens):
                            reason = "length"
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
            finish_reasons.append(reason)

    return texts, finish_reasons
