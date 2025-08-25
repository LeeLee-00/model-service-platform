"""This module provides application settings and configuration for the model service."""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch
from app.core.logger import setup_logger

logger = setup_logger("model_service.config")


class Settings(BaseSettings):
    """Application settings and configuration"""

    # Model settings
    MODEL_NAME: str = ""
    MODEL_TYPE: str = "LLM"
    MODELS_DIR: str = "/models"
    TRANSFORMERS_OFFLINE: bool = True

    # Batch processing settings
    TORCH_DEVICE: Optional[str] = None
    BATCH_SIZE: int = 1
    BATCH_TIMEOUT: float = 0.01
    MAX_LENGTH: int = 50
    TEMPERATURE: float = 0.7
    LOG_LEVEL: str = "DEBUG"

    # Aggregation Knobs let you tune between granularity and latency per model.
    MIN_TOKENS: int = 15
    MAX_CHARS: int = 128
    SENTENCE_END: set[str] = {".", "!", "?", "\n"}
    WHITESPACE_CHUNK_CHARS: int = 24

    # API settings
    API_TITLE: str = "Model Service API"
    API_DESCRIPTION: str = "LLM service"
    API_VERSION: str = "1.0.0"

    # HF Token
    HF_TOKEN: str = ""

    # Min.io Connectivity
    MINIO_SVC_USER: str = "admin"
    MINIO_BUCKET: str = "models"
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_SVC_PASSWORD: str = "password"

    # pylint: disable=invalid-name
    @property
    def DEVICE(self) -> torch.device:
        """Pick the best available torch device (override with TORCH_DEVICE)."""
        if self.TORCH_DEVICE:
            forced = self.TORCH_DEVICE.lower()
            if forced == "cuda" and torch.cuda.is_available():
                logger.info("Using forced CUDA device")
                return torch.device("cuda")
            if forced == "mps" and torch.backends.mps.is_available():
                logger.info("Using forced MPS device")
                return torch.device("mps")
            logger.info("Falling back to CPU device")
            return torch.device("cpu")

        # Auto-detect if no override given
        if torch.cuda.is_available():
            logger.info("Auto-detected CUDA device")
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            logger.info("Auto-detected MPS device")
            return torch.device("mps")
        logger.info("Using CPU device")
        return torch.device("cpu")

    # pylint: disable=invalid-name
    @property
    def MODEL_HF_DIR_NAME(self) -> str:
        """Return the standardized model directory name used by HuggingFace cache."""
        model_name = self.MODEL_NAME.replace("/", "--")
        return "models--" + model_name

    model_config = SettingsConfigDict(env_prefix="", env_file=None, extra="allow")


settings = Settings()
