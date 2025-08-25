# LLM Service

A production-ready FastAPI microservice for serving Large Language Models (LLMs) with OpenAI-compatible APIs.  
Supports both streaming and non-streaming chat completions, batching, stop sequences, token usage reporting, and efficient model management.

---

## 🚀 Features

- **OpenAI-compatible** `/v1/chat/completions` endpoint (streaming & non-streaming)
- **Batching** for high throughput and efficient GPU/CPU utilization
- **Stop sequence** support (token-level and text-level)
- **Token usage** reporting (prompt, completion, total)
- **Streaming** responses via Server-Sent Events (SSE)
- **MinIO/S3** integration for model storage and sharing
- **Extensible**: add new models or endpoints easily

---

## 🏗️ Project Structure

```
llm/
├── handlers/
│   ├── non_streaming.py      # Non-streaming chat completion handler
│   └── streaming.py          # Streaming chat completion handler
├── models.py                 # Pydantic models and stopping criteria
├── pipe.py                   # Pipeline initialization and management
├── routes.py                 # FastAPI routes for chat completions
├── service.py                # Background batching worker for non-streaming
├── util.py                   # Shared utilities (prompt, stops, config, etc.)
└── testing/
    └── locustfile.py         # Load testing with Locust
```

---

## ⚡️ Quickstart

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**  
   Set environment variables or edit your `.env` file (see `.env.template`).

3. **Run the service**  
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

4. **Send a request**  
   ```bash
   curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [{"role": "user", "content": "Hello!"}],
       "max_tokens": 32,
       "stream": false
     }'
   ```

---

## 🛣️ API Endpoints

### `POST /v1/chat/completions`

OpenAI-compatible chat completions.

**Request:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a joke."}
  ],
  "max_tokens": 32,
  "temperature": 0.7,
  "top_p": 0.9,
  "n": 1,
  "stream": false,
  "stop": ["\nUser:", "\nAssistant:"]
}
```

**Response (non-streaming):**
```json
{
  "id": "chatcmpl-xxxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "your-model-name",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Why did the chicken..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 20,
    "total_tokens": 32
  }
}
```

**Response (streaming):**  
Server-Sent Events (SSE) with incremental `chat.completion.chunk` objects.

---

## ⚙️ How It Works

- **API Layer**: Receives requests, builds prompts, normalizes stops, and delegates to streaming or batching handlers.
- **Batching Worker**: Collects and groups compatible requests, runs batched inference, and returns results.
- **Streaming Handler**: Uses Hugging Face's `TextIteratorStreamer` for low-latency token streaming.
- **Pipeline Management**: Ensures a single, async-safe Hugging Face pipeline instance.
- **Utilities**: Shared helpers for prompt construction, stop handling, and output post-processing.

---

## 🧪 Testing

- Use [Locust](https://locust.io/) for load testing:
  ```bash
  locust -f app/llm/testing/locustfile.py
  ```

---

## 📝 Extending

- **Add new endpoints**: Create new route modules and register them in `main.py`.
- **Swap models**: Change model config in `.env` or via environment variables.
- **Customize batching**: Tune `BATCH_SIZE` and `BATCH_TIMEOUT` in your config.

---

## 📂 Related Files

- [`models.py`](models.py): Pydantic request/response models, stopping criteria
- [`routes.py`](routes.py): FastAPI endpoints
- [`service.py`](service.py): Batching worker loop
- [`handlers/`](handlers/): Streaming and non-streaming handlers
- [`pipe.py`](pipe.py): Pipeline lifecycle management
- [`util.py`](util.py): Prompt, stop, and config utilities

---