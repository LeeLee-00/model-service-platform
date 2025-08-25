"""Data models used by the LLM service.

This module defines:
- Pydantic request/DTO models for OpenAI-compatible chat completions.
- A custom `StoppingCriteria` implementation for token-sequence stops.
"""

from typing import Any, List, Literal, Optional

import torch
from pydantic import BaseModel, Field
from transformers import StoppingCriteria


class StreamOptions(BaseModel):  # pylint: disable=too-few-public-methods
    """Optional streaming configuration for OpenAI-compatible responses.

    Attributes
    ----------
    include_usage : Optional[bool]
        When True, include `usage` (token counts) in the final streaming chunk.
    """

    include_usage: Optional[bool] = None


class ChatMessage(BaseModel):  # pylint: disable=too-few-public-methods
    """A single chat turn message in role/content format.

    Attributes
    ----------
    role : Literal["system", "user", "assistant"]
        The role of the speaker for this message.
    content : str
        The textual content of the message.
    """

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):  # pylint: disable=too-few-public-methods
    """OpenAI-compatible chat completion request payload.

    Attributes
    ----------
    messages : List[ChatMessage]
        Ordered list of chat messages (context + latest user query).
    max_tokens : Optional[int]
        Upper bound on number of new tokens to generate.
    temperature : Optional[float]
        Sampling temperature; 0 or None implies greedy decoding.
    top_p : Optional[float]
        Nucleus sampling probability (used only when sampling).
    n : Optional[int]
        Number of completions to generate (non-streaming supports >1).
    stream : Optional[bool]
        When True, use Server-Sent Events for incremental output.
    stop : Optional[Any]
        String or list of strings to stop generation on.
    stream_options : Optional[StreamOptions]
        Additional streaming options (e.g., include_usage).
    """
    model: Optional[str] = None  # model name is ignored
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0.0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    n: Optional[int] = Field(default=None, ge=1, le=8)
    stream: Optional[bool] = False
    stop: Optional[Any] = None  # str | List[str]
    stream_options: Optional[StreamOptions] = None


class StopOnTokens(StoppingCriteria):  # pylint: disable=too-few-public-methods
    """Stopping criteria that halts when any stop token-id sequence matches."""

    def __init__(self, stop_sequences: list[list[int]]) -> None:
        """Initialize with a list of non-empty stop token-id sequences."""
        super().__init__()
        self.stop_sequences = [seq for seq in stop_sequences if seq]

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,  # noqa: ARG002 (signature required by HF)
        **kwargs: Any,  # noqa: ANN003
    ) -> bool:
        """Return True if the generated ids end with any stop sequence."""
        if input_ids is None or input_ids.shape[0] == 0:
            return False
        ids = input_ids[0].tolist()
        for seq in self.stop_sequences:
            seq_len = len(seq)  # renamed from 'L' to satisfy pylint C0103
            if seq_len and len(ids) >= seq_len and ids[-seq_len:] == seq:
                return True
        return False
