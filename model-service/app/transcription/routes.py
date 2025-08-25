"""Routes for Whisper speech-to-text transcription service."""

import asyncio
from typing import Annotated, Optional
from fastapi import APIRouter, Depends, File, Form, UploadFile, Request
from fastapi.responses import JSONResponse

from app.core.logger import setup_logger

logger = setup_logger("model_service.transcription.routes")

router = APIRouter()


def get_model_queue(request: Request) -> asyncio.Queue:
    """Dependency to get the model queue from the application state."""
    return request.app.state.model_queue


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    model_queue: Annotated[asyncio.Queue, Depends(get_model_queue)],
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    """Transcribes audio into the input language."""
    if not file.filename.lower().endswith(
        (".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm")
    ):
        return JSONResponse(
            {
                "error": "Unsupported file type. Must be mp3, mp4, mpeg, mpga, m4a, wav, or webm"
            },
            status_code=400,
        )

    try:
        content = await file.read()
        response_q = asyncio.Queue()
        # Correct tuple syntax
        await model_queue.put((content, language, response_q))

        result = await response_q.get()

        if "error" in result:
            return JSONResponse({"error": result["error"]}, status_code=500)

        return result

    except (OSError, ValueError, asyncio.TimeoutError) as e:
        logger.error("Error processing audio file: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)
