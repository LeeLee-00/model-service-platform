# Model Service Platform

A **containerized, multi-GPU AI inference platform** for serving Hugging Face models with OpenAI-compatible APIs, unified storage, and intelligent routing. Purpose-built for **offline/air-gapped deployments** with multiple GPUs and diverse model types.

---

![Model Services High-Level Architecture](docs/Model-Service-Diagram.png)
---

## Features

- **Multi-GPU Orchestration:** Dedicated GPU per service with intelligent resource allocation
- **Offline-First Architecture:** Pre-download models to MinIO for air-gapped operation
- **Service Discovery & Load Balancing:** Automatic health monitoring and intelligent request routing
- **Multi-Model Serving:** Run LLMs, embeddings, vision, and transcription models simultaneously
- **OpenAI-Compatible API:** Full compatibility with OpenAI spec for easy integration
- **Unified Gateway:** Single entry point for all model services with automatic failover
- **Model Registry:** Centralized model management, versioning, and lifecycle control
- **Production-Ready:** Request batching, GPU monitoring, and observability built-in

---

## Use Cases

**Perfect For:**
- 🔒 Air-gapped/offline deployments (military, healthcare, finance)
- 🎮 Multi-GPU clusters requiring heterogeneous workloads
- 🏢 Organizations needing centralized model management
- 🔬 Research teams running multiple model types concurrently

**Not Ideal For:**
- Single model, single GPU setups (use [Ollama](https://ollama.ai/) instead)
- Cloud-hosted with internet access (use hosted APIs)
- Prototyping/learning (too much infrastructure overhead)

---

## OpenAI API Standard

This codebase follows the `v1` specification from OpenAI on API connectivity with chat models. As a result, off-the-shelf and open source tools designed
to interact with OpenAI models (such as HuggingFace's ChatUI) can easily integrate with this codebase. This also enables future data challenge
participants and remote data team developers to develop toolbox applications with their own OpenAI available models. For more information on OpenAI's
platform standards, view [their reference guide](https://platform.openai.com/docs/overview).

---

## Architecture Overview

### Standard Mode (Development/Single GPU)
```
[Chat UI] <--> [Model Services] <--> [MinIO Storage] <--> [Hugging Face Hub]
```

### GPU Cluster Mode (Production/Multi-GPU)
```
                                    ┌─> [LLM Primary - GPU 0]
                                    │
[Chat UI] <--> [Gateway] ---------> ├─> [LLM Secondary - GPU 1]
                ↕                   │
         [Service Registry]         ├─> [Multimodal - GPU 2]
                ↕                   │
         [Model Registry]           └─> [Embedding - GPU 3]
                ↕
           [MinIO Storage] <--> [Hugging Face Hub]
```

**Components:**
- **Gateway:** Unified API entry point with intelligent routing and load balancing
- **Service Registry:** Health monitoring, service discovery, and GPU statistics
- **Model Registry:** Centralized model management (download, list, delete from MinIO)
- **Model Services:** GPU-bound workers for different model types
- **MinIO:** Offline model storage (S3-compatible)
- **Chat UI:** Web interface for end users

---

## Quick Start

### Standard Mode (Development)

For development or single-GPU setups:

```sh
# 1. Clone and configure
git clone https://gitlab.wildfireworkspace.com/eop/data-toolbox/model-service.git
cd model-service
cp .env.example .env
# Edit .env with your credentials

# 2. Launch services
docker compose up -d

# 3. (Optional) Add Chat UI
cp chatui.env.template chatui.env
docker compose -f docker-compose.yml -f docker-compose.chatui.yml up -d
```

### GPU Cluster Mode (Production)

For multi-GPU offline deployments:

#### Step 1: Pre-download Models (Internet Required)

```sh
# Run this ONCE when you have internet connectivity
./scripts/offline-setup.sh
```

This will:
- Start MinIO and Registry services
- Download models from Hugging Face to MinIO
- Prepare for offline operation

Monitor progress:
```sh
docker compose logs -f registry
```

Verify models downloaded:
```sh
curl http://localhost:8090/models | jq
```

#### Step 2: Deploy GPU Cluster (Offline)

```sh
# Start all services with GPU assignments
docker compose -f docker-compose.gpu-cluster.yml up -d
```

This will:
- Assign each service to a dedicated GPU
- Start Gateway for unified API access
- Enable service discovery and health monitoring

#### Step 3: Verify Cluster Health

```sh
# Check all services
curl http://localhost:8080/services | jq

# Check GPU utilization
curl http://localhost:8080/gpu-stats | jq

# Test chat completion
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

---

## API Endpoints

### Gateway (Port 8080)
- `GET /services` - Service health and status
- `GET /gpu-stats` - Cluster-wide GPU statistics
- `POST /v1/chat/completions` - Unified chat endpoint (auto-routes)
- `POST /v1/embeddings` - Unified embeddings endpoint
- `GET /v1/models` - List all available models
- `GET /health` - Gateway health check

### Model Registry (Port 8090)
- `GET /models` - List models in MinIO
- `POST /models/{name}/download` - Download model from HuggingFace
- `GET /models/{name}` - Get model info
- `DELETE /models/{name}` - Remove model from storage

### Individual Services (Ports 8000-8004)
- `POST /v1/chat/completions` - Direct model access
- `GET /v1/models` - Model information
- `GET /health` - Service health
- `GET /gpu-stats` - Service GPU stats

---

## GPU Resource Management

The GPU cluster mode assigns specific GPUs to services:

| Service | GPU ID | Model | Use Case |
|---------|--------|-------|----------|
| llm-primary | 0 | Qwen 0.5B | Fast general-purpose chat |
| llm-secondary | 1 | TinyLlama 1.1B | Backup/specialized tasks |
| multimodal | 2 | LLaVA 7B | Vision + text understanding |
| embedding | 3 | all-MiniLM-L6 | High-throughput embeddings |
| transcription* | 4 | Whisper Base | Speech-to-text (optional) |

**Customize GPU assignments** in `docker-compose.gpu-cluster.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0']  # Change GPU ID here
          capabilities: [gpu]
```

---

## Documentation

- **[Docker Compose Commands Reference](docs/DOCKER-COMPOSE-COMMANDS.md)** - Complete guide for managing different stacks
- **[GPU Cluster Architecture Guide](docs/GPU-CLUSTER-GUIDE.md)** - Deep dive into multi-GPU setup
- **[Alternatives Comparison](docs/ALTERNATIVES-COMPARISON.md)** - vs Ollama, vLLM, LocalAI, etc.

---

## Common Tasks

### Managing Services

**Start/stop different stacks:**
```sh
# GPU Cluster
docker compose -f docker-compose.yml -f docker-compose.gpu-cluster.yml up -d
docker compose -f docker-compose.yml -f docker-compose.gpu-cluster.yml down

# ChatUI
docker compose -f docker-compose.yml -f docker-compose.chatui.yml up -d
docker compose -f docker-compose.yml -f docker-compose.chatui.yml down

# Development
docker compose up -d
docker compose down
```

See [DOCKER-COMPOSE-COMMANDS.md](docs/DOCKER-COMPOSE-COMMANDS.md) for complete reference.

### Monitoring

**View logs:**
```sh
# All services
docker compose logs -f

# Specific service
docker compose logs -f gateway
docker logs model-gateway --tail=50

# Check status
docker ps
docker stats
```

**Health checks:**
```sh
# Gateway
curl http://localhost:8080/health
curl http://localhost:8080/services | jq

# Registry
curl http://localhost:8090/health
curl http://localhost:8090/models | jq

# Individual services
curl http://localhost:8000/health
```

---

## Project Workflow

### Development Cycle

1. **Start Services:** Launch with `docker compose up -d`
2. **Check Logs:** Monitor startup with `docker compose logs -f [service-name]`
3. **Test APIs:** Use the interactive docs at `http://localhost:8000/docs`
4. **Make Changes:** Edit code, then rebuild: `docker compose up -d --build [service-name]`
5. **Stop Services:** `docker compose down` (add `-v` to remove volumes)

### Adding/Updating Models

1. **Edit Configuration:**
   - Add new service in `docker-compose.yml`
   - Update `chatui.env` to expose it in the UI

2. **Rebuild and Restart:**
   ```sh
   docker compose up -d --build
   ```

3. **Verify:**
   - Check API: `curl http://localhost:8000/v1/models`
   - Check Chat UI: Visit http://localhost:3000/settings

### Chat UI Configuration

The Chat UI uses version **0.8.4** (pinned) to ensure stable behavior with custom models. 

**Why version 0.8.4?**
- Newer versions (>= 0.9.0) auto-fetch 110+ community models from Hugging Face
- Version 0.8.4 respects your custom `MODELS` configuration
- Recommended for self-hosted deployments with custom endpoints

**To customize models:**
1. Edit `chatui.env`
2. Modify the `MODELS` array with your endpoints
3. Restart: `docker compose -f docker-compose.yml -f docker-compose.chatui.yml restart chatui`

Example multi-model configuration:
```env
MODELS=[{"name":"qwen","displayName":"Qwen 0.5B","endpoints":[{"type":"openai","baseURL":"http://llm:8000/v1"}]},{"name":"llava","displayName":"LLaVA Vision","multimodal":true,"endpoints":[{"type":"openai","baseURL":"http://multimodal:8003/v1"}]}]
```

---

## Adding a New Model

1. **Add Service to `docker-compose.yml`:**
   ```yaml
   my-new-model:
     build: ./model-service
     environment:
       MODEL_NAME: "huggingface/my-model"
       MODEL_TYPE: "llm"
     ports:
       - "8005:8000"
   ```

2. **Update Chat UI Configuration (optional):**
   Edit `chatui.env` and add your model to the `MODELS` array:
   ```env
   MODELS=[...,{"name":"my-model","displayName":"My Model","endpoints":[{"type":"openai","baseURL":"http://my-new-model:8000/v1"}]}]
   ```

3. **Restart Services:**
   ```sh
   docker compose up -d --build
   ```

No code changes are needed!

---

## Services

| Service         | Description                        | Default Port |
|-----------------|------------------------------------|--------------|
| minio           | S3-compatible object storage        | 9000/9001    |
| chatui          | Web chat interface                 | 3000         |
| llm             | LLM (e.g., Qwen)                   | 8000         |
| tinyllama       | LLM (e.g., TinyLlama)              | 8001         |
| embedding       | Embedding model                    | 8002         |
| multimodal      | Multimodal (image+text) model      | 8003         |
| transcription   | Speech-to-text model               | 8004         |

---

## Environment Variables

### Core Services (`.env` file)

See `.env.example` for all required variables:
- `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD` - MinIO admin credentials
- `MINIO_SVC_USER`, `MINIO_SVC_PASSWORD` - Service account credentials
- `MINIO_ENDPOINT` - MinIO server endpoint
- `HF_TOKEN` - Hugging Face access token for downloading models

### Chat UI (`chatui.env` file)

**Important:** Copy `chatui.env.template` to `chatui.env` before first use.

Key variables:
- `MONGODB_URL` - MongoDB connection string
- `MODELS` - JSON array of model configurations
- `PUBLIC_APP_NAME` - Custom branding for the UI
- `ENABLE_COMMUNITY_MODELS` - Set to `false` to only show your custom models

---

## Troubleshooting

### Chat UI shows 110+ models instead of my custom ones

**Solution:** Ensure you're using Chat UI version **0.8.4** (check `docker-compose.chatui.yml`). Newer versions auto-fetch community models by default.

### Model service won't start

**Check:**
1. MinIO is running: `docker compose ps minio`
2. Environment variables are set in `.env`
3. Hugging Face token has correct permissions
4. Logs: `docker compose logs [service-name]`

### Chat UI can't connect to models

**Verify:**
1. Model services are running: `docker compose ps`
2. `chatui.env` has correct `baseURL` (use Docker service names like `http://llm:8000/v1`)
3. All services are on the same Docker network

### GPU Not Detected

**Check:**
1. NVIDIA drivers installed: `nvidia-smi`
2. NVIDIA Container Toolkit installed
3. Test GPU access: `docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi`
4. Check docker-compose.gpu-cluster.yml has correct `device_ids`

### Port Already in Use

**Fix:**
```sh
# Find what's using the port
sudo lsof -i :8080

# Stop conflicting services
docker compose down
```

**For more troubleshooting help, see [DOCKER-COMPOSE-COMMANDS.md](docs/DOCKER-COMPOSE-COMMANDS.md#troubleshooting)**

---

## License

MIT License

---

## Acknowledgements

- [Hugging Face](https://huggingface.co/)