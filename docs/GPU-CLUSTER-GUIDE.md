# GPU Cluster Quick Reference

## Architecture Components

```
┌─────────────┐
│  Chat UI    │ (Port 3000) - Web interface
│  (Optional) │
└──────┬──────┘
       │
┌──────▼──────┐
│   Gateway   │ (Port 8080) - Unified API + Routing
└──────┬──────┘
       │
       ├─────────────────────┐
       │                     │
┌──────▼──────┐     ┌────────▼────────┐
│  Services   │     │ Model Registry  │ (Port 8090)
│  Discovery  │     │  (Download/     │
│  & Health   │     │   Management)   │
└──────┬──────┘     └────────┬────────┘
       │                     │
       ├─────────────────────┘
       │
┌──────▼─────────────────────────────┐
│         Model Services             │
│  ┌────────┬────────┬────────┐     │
│  │ GPU 0  │ GPU 1  │ GPU 2  │...  │
│  │  LLM1  │  LLM2  │ Vision │     │
│  └────────┴────────┴────────┘     │
└────────────────┬───────────────────┘
                 │
         ┌───────▼────────┐
         │  MinIO Storage │ (Ports 9000/9001)
         │  (S3-compat)   │
         └────────────────┘
```

## Common Commands

### Offline Setup (Run Once with Internet)
```bash
# Download all models to MinIO
./scripts/offline-setup.sh

# Monitor downloads
docker compose logs -f registry

# Verify models
curl http://localhost:8090/models | jq
```

### Start GPU Cluster (Offline Mode)
```bash
# Full cluster
docker compose -f docker-compose.gpu-cluster.yml up -d

# Check status
docker compose -f docker-compose.gpu-cluster.yml ps

# View logs
docker compose -f docker-compose.gpu-cluster.yml logs -f [service-name]

# Stop cluster
docker compose -f docker-compose.gpu-cluster.yml down
```

### Health Checks
```bash
# All services health
curl http://localhost:8080/services | jq

# GPU stats (cluster-wide)
curl http://localhost:8080/gpu-stats | jq

# Individual service
curl http://localhost:8000/health | jq
curl http://localhost:8000/gpu-stats | jq
```

### Model Management
```bash
# List models in MinIO
curl http://localhost:8090/models | jq

# Download new model (requires internet)
curl -X POST http://localhost:8090/models/my-model/download \
  -H "Content-Type: application/json" \
  -d '{"hf_repo": "meta-llama/Llama-3.2-1B-Instruct"}'

# Get model details
curl http://localhost:8090/models/qwen | jq

# Delete model
curl -X DELETE http://localhost:8090/models/qwen
```

### API Usage

#### Via Gateway (Recommended)
```bash
# Chat completion (auto-routes to best LLM)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is AI?"}],
    "temperature": 0.7,
    "max_tokens": 512
  }'

# Embeddings
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world"}'

# List all models
curl http://localhost:8080/v1/models | jq
```

#### Direct Service Access
```bash
# Talk to specific LLM
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Get embeddings
curl -X POST http://localhost:8002/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "test"}'
```

## Service Ports

| Service | Port | Purpose |
|---------|------|---------|
| Gateway | 8080 | Unified API entry |
| Registry | 8090 | Model management |
| LLM Primary | 8000 | Qwen (GPU 0) |
| LLM Secondary | 8001 | TinyLlama (GPU 1) |
| Embedding | 8002 | Embeddings (GPU 3) |
| Multimodal | 8003 | LLaVA Vision (GPU 2) |
| Transcription | 8004 | Whisper (GPU 4, optional) |
| MinIO API | 9000 | S3-compatible storage |
| MinIO Console | 9001 | Web UI |
| Chat UI | 3000 | User interface |

## GPU Assignments

Edit `docker-compose.gpu-cluster.yml` to customize:

```yaml
services:
  llm-primary:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']  # Change to your GPU ID
              capabilities: [gpu]
    environment:
      CUDA_VISIBLE_DEVICES: "0"  # Container sees it as GPU 0
```

## Troubleshooting

### Service Won't Start
```bash
# Check logs
docker compose -f docker-compose.gpu-cluster.yml logs [service-name]

# Check GPU availability
nvidia-smi

# Verify MinIO is running
docker compose ps minio
```

### Model Not Found
```bash
# Check MinIO
curl http://localhost:8090/models | jq

# Re-download
./scripts/offline-setup.sh
```

### Gateway Not Routing
```bash
# Check service health
curl http://localhost:8080/services | jq

# Check if backend services are up
docker compose -f docker-compose.gpu-cluster.yml ps
```

### Out of GPU Memory
```bash
# Check GPU usage
curl http://localhost:8080/gpu-stats | jq

# Restart service to clear memory
docker compose -f docker-compose.gpu-cluster.yml restart llm-primary
```

## Performance Tips

1. **Batch Requests**: Use higher `MAX_BATCH_SIZE` for embeddings (they're smaller)
2. **Model Selection**: Use Qwen (GPU 0) for speed, TinyLlama (GPU 1) for longer context
3. **Memory Management**: Monitor with `/gpu-stats` endpoint
4. **Load Distribution**: Gateway automatically routes to fastest available service
5. **Caching**: MinIO stores models persistently - no re-download on restart

## Chat UI Integration

Update `chatui.env` to use Gateway:

```env
MODELS=[{
  "name": "cluster",
  "displayName": "Multi-GPU Cluster",
  "endpoints": [{
    "type": "openai",
    "baseURL": "http://gateway:8080/v1"
  }]
}]
```

Or direct service access:

```env
MODELS=[
  {
    "name": "qwen",
    "displayName": "Qwen (Fast)",
    "endpoints": [{"type": "openai", "baseURL": "http://llm-primary:8000/v1"}]
  },
  {
    "name": "llava",
    "displayName": "LLaVA Vision",
    "multimodal": true,
    "endpoints": [{"type": "openai", "baseURL": "http://multimodal:8003/v1"}]
  }
]
```

## Monitoring

### Docker Stats
```bash
docker stats
```

### GPU Utilization
```bash
watch -n 1 nvidia-smi
```

### Service Health Dashboard
```bash
watch -n 2 'curl -s http://localhost:8080/services | jq ".services | to_entries | .[] | {name: .key, healthy: .value.healthy, response_ms: .value.response_time_ms}"'
```

### GPU Memory Dashboard
```bash
watch -n 2 'curl -s http://localhost:8080/gpu-stats | jq ".services"'
```
