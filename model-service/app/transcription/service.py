"""Transcription service using Hugging Face Transformers pipeline."""

import asyncio
import tempfile

from transformers import pipeline

from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger("model_service.whisper.service")


async def model_server_loop(queue: asyncio.Queue):
    """Main model server loop for processing speech recognition requests."""
    try:
        pipe = pipeline(
            task="automatic-speech-recognition",
            model=f"{settings.MODELS_DIR}/{settings.MODEL_HF_DIR_NAME}",
            device=settings.DEVICE,
            return_timestamps=True,
        )
        logger.info("Speech Recognition Pipeline initialized successfully")
    except (ValueError, OSError) as exc:
        logger.error("Error initializing pipeline: %s", exc)
        raise

    while True:
        audio_bytes, language, response_q = await queue.get()
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio.flush()

                # Use Whisper's native long-form transcription
                generate_kwargs = {"task": "transcribe"}
                if language:
                    generate_kwargs["language"] = language

                transcription = pipe(temp_audio.name, generate_kwargs=generate_kwargs)

            await response_q.put(
                {
                    "text": transcription["text"],
                    "model": settings.MODEL_NAME,
                    "task": "transcribe",
                    "language": language or "auto",
                }
            )
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error("Error processing speech request: %s", exc)
            await response_q.put({"error": str(exc)})
