"""Pydantic models for multimodal requests and responses."""

from typing import List, Optional, Union
from pydantic import BaseModel, Field

from app.llm.models import StreamOptions


class ImageContent(BaseModel):
    """Image content with base64 data."""

    type: str  # "image"
    image: str  # base64 encoded image data


class TextContent(BaseModel):
    """Text content."""

    type: str  # "text"
    text: str


class MultimodalMessage(BaseModel):
    """Individual message in a conversation."""

    role: str  # "user", "assistant", "system"
    content: Union[str, List[Union[TextContent, ImageContent]]]


class MultimodalRequest(BaseModel):
    """Request model for multimodal chat completion."""

    messages: List[MultimodalMessage]
    model: Optional[str] = None
    max_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    stream_options: Optional[StreamOptions] = None


class MultimodalChoice(BaseModel):
    """Individual choice in the response."""

    index: int
    message: MultimodalMessage
    finish_reason: Optional[str] = "stop"


class MultimodalUsage(BaseModel):
    """Usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class MultimodalResponse(BaseModel):
    """Response model for multimodal chat completion."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[MultimodalChoice]
    usage: MultimodalUsage
