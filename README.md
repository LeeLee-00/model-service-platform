# Model Service Platform

A **containerized, multi-model AI inference platform** for serving Hugging Face models with OpenAI-compatible APIs, unified storage, and a modern web UI. Easily deploy LLMs, embedding models, multimodal models, and more—locally or in production.

---

![Model Services High-Level Architecture](docs/Model-Service-Diagram.png)
---

## Features

- **Multi-Model Serving:** Run multiple LLMs, embedding, multimodal, and transcription models in parallel, each as a separate service.
- **OpenAI-Compatible API:** Each model exposes `/v1/chat/completions` and related endpoints, supporting both streaming and non-streaming requests.
- **Unified Object Storage:** Uses MinIO (S3-compatible) for model and artifact storage, enabling fast startup and easy model management.
- **Web Chat UI:** Modern chat interface connects to any deployed model via OpenAI-compatible endpoints.
- **Easy Extensibility:** Add new models or services by editing `docker-compose.yml`—no code changes required.
- **Efficient Inference:** Request batching, token usage reporting, and prompt formatting for production workloads.
- **Development Friendly:** Hot-reload, interactive API docs, and CLI tools for local development.

---

## OpenAI API Standard

This codebase follows the `v1` specification from OpenAI on API connectivity with chat models. As a result, off-the-shelf and open source tools designed
to interact with OpenAI models (such as HuggingFace's ChatUI) can easily integrate with this codebase. This also enables future data challenge
participants and remote data team developers to develop toolbox applications with their own OpenAI available models. For more information on OpenAI's
platform standards, view [their reference guide](https://platform.openai.com/docs/overview).

---

## Architecture Overview

```
[Chat UI] <--> [Model Services: LLM | Embedding | Multimodal | Transcription] <--> [MinIO Storage] <--> [Hugging Face Hub]
```

- **MinIO:** Central S3-compatible storage for all models and artifacts.
- **Model Services:** Each service runs a different Hugging Face model, exposes OpenAI-compatible APIs, and shares storage.
- **Chat UI:** Web app for interacting with any deployed model.
- **Docker Compose:** Orchestrates all services and networking.

---

## Quick Start

### 1. Clone the Repository

```sh
git clone https://github.com/LeeLee-00/model-service-platform.git
cd model-service
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in required values (MinIO credentials, Hugging Face token, etc.).

### 3. Launch the Platform

```sh
docker compose up --build
```

- Model APIs are available at ports `8000`, `8001`, etc.

### 4. Utilizing the ChatUI

Use the provided override to add the chat UI to the deployment.

```sh
docker compose -f docker-compose.yml -f docker-compose.chatui.yml up -d
```

- Access the **Chat UI** at [http://localhost:3000](http://localhost:3000)

---

## Adding a New Model

1. Duplicate a service block in `docker-compose.yml`.
2. Set `MODEL_NAME` to your desired Hugging Face model.
3. Optionally, update the Chat UI `MODELS` environment variable to add it to the UI.

No code changes are needed!

---

## Services

| Service         | Description                        | Default Port |
|-----------------|------------------------------------|--------------|
| minio           | S3-compatible object storage        | 9000/9001    |
| chatui          | Web chat interface                 | 4000         |
| llm             | LLM (e.g., Qwen)                   | 8000         |
| tinyllama       | LLM (e.g., TinyLlama)              | 8001         |
| embedding       | Embedding model                    | 8002         |
| multimodal      | Multimodal (image+text) model      | 8003         |
| transcription   | Speech-to-text model               | 8004         |

---

## Environment Variables

See `.env.example` for all required variables:
- `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`
- `MINIO_SVC_USER`, `MINIO_SVC_PASSWORD`
- `MINIO_ENDPOINT`
- `HF_TOKEN` (Hugging Face access token)

---

## License

MIT License

---

## Acknowledgements

- [Hugging Face](https://huggingface.co/)
