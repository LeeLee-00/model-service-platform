"""Utils for handling multimodal actions"""

import base64
import io

from PIL import Image

from app.core.logger import setup_logger

logger = setup_logger("model_service.multimodal.util")


def process_image_content(content_item):
    """Process image content from base64 for batch processing.

    Args:
        content_item: Dictionary with type and image data

    Returns:
        Processed content item with PIL Image object

    Raises:
        ValueError: If image processing fails
    """
    if content_item.get("type") == "image":
        if "image" in content_item:
            # Handle base64 image
            try:
                image_data = base64.b64decode(content_item["image"])
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                return {"type": "image", "image": image}
            except Exception as exc:
                logger.error("Failed to decode base64 image: %s", exc)
                raise ValueError(f"Invalid base64 image: {exc}") from exc
        elif "url" in content_item:
            # For future URL support, but not implemented in offline environment
            raise ValueError("URL images not supported in offline environment")
    return content_item


def prepare_multimodal_messages(messages):
    """Convert multimodal messages to the format expected by the pipeline for batching.

    Args:
        messages: List of MultimodalMessage Pydantic objects

    Returns:
        List of processed messages with decoded images
    """
    processed_messages = []

    for message in messages:
        # Handle Pydantic MultimodalMessage objects
        processed_message = {"role": message.role}

        if isinstance(message.content, str):
            # Simple text content
            processed_message["content"] = [{"type": "text", "text": message.content}]
        elif isinstance(message.content, list):
            # Mixed content with text and images
            processed_content = []
            for content_item in message.content:
                if hasattr(content_item, "type"):  # Pydantic object
                    if content_item.type == "image":
                        processed_content.append(
                            process_image_content(
                                {"type": "image", "image": content_item.image}
                            )
                        )
                    else:
                        processed_content.append(
                            {
                                "type": content_item.type,
                                "text": getattr(content_item, "text", ""),
                            }
                        )
                else:
                    processed_content.append(content_item)
            processed_message["content"] = processed_content
        else:
            processed_message["content"] = [
                {"type": "text", "text": str(message.content)}
            ]

        processed_messages.append(processed_message)

    return processed_messages
