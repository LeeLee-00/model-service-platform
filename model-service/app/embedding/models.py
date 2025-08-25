"""Pydantic models for embedding requests and responses."""

from typing import Union, List

from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""

    input: Union[str, List[str]]
    model: str = None


class EmbeddingUsage(BaseModel):
    """Usage statistics for embedding generation."""

    prompt_tokens: int
    total_tokens: int


class EmbeddingData(BaseModel):
    """Data model for individual embedding results."""

    object: str = "embedding"
    embedding: List[List[float]]
    index: int


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""

    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage
