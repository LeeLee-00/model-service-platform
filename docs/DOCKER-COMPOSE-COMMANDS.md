# Docker Compose Commands Reference

This guide provides commands for managing different service stacks in this project.

## Table of Contents
- [Prerequisites](#prerequisites)
- [GPU Cluster Stack](#gpu-cluster-stack)
- [ChatUI Stack](#chatui-stack)
- [Development Stack](#development-stack)
- [Common Operations](#common-operations)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before running any stack, ensure you have:
- Docker and Docker Compose installed
- `.env` file configured (copy from `.env.example` if available)
- For ChatUI: `chatui.env` file configured (copy from `chatui.env.template`)

---

## GPU Cluster Stack

**Use Case:** Production multi-GPU deployment with Gateway, Registry, and model services

**Services Included:**
- Gateway (port 8080) - API routing and service discovery
- Registry (port 8090) - Model management
- MinIO (port 9000) - S3-compatible storage
- 4x Model Services (ports 8000-8003) - GPU workload distribution

### Commands

```bash
# Start all GPU cluster services
docker compose -f docker-compose.yml -f docker-compose.gpu-cluster.yml up -d

# View logs (all services)
docker compose -f docker-compose.yml -f docker-compose.gpu-cluster.yml logs -f

# View logs (specific service)
docker compose -f docker-compose.yml -f docker-compose.gpu-cluster.yml logs -f gateway
docker compose -f docker-compose.yml -f docker-compose.gpu-cluster.yml logs -f registry

# Rebuild specific service
docker compose -f docker-compose.yml -f docker-compose.gpu-cluster.yml build gateway
docker compose -f docker-compose.yml -f docker-compose.gpu-cluster.yml up -d gateway

# Stop all services
docker compose -f docker-compose.yml -f docker-compose.gpu-cluster.yml down

# Stop and remove volumes (WARNING: deletes data)
docker compose -f docker-compose.yml -f docker-compose.gpu-cluster.yml down -v
```

### Testing Endpoints

```bash
# Gateway health
curl http://localhost:8080/health

# Gateway service discovery
curl http://localhost:8080/services

# Registry health
curl http://localhost:8090/health

# List models in registry
curl http://localhost:8090/models
```

---

## ChatUI Stack

**Use Case:** Web-based chat interface for interacting with models

**Services Included:**
- Chat UI (port 3000) - Web interface
- MongoDB (port 27017) - Conversation persistence
- MinIO (port 9000) - S3-compatible storage
- Model Service (port 8000) - Backend LLM service

### Commands

```bash
# Start ChatUI stack
docker compose -f docker-compose.yml -f docker-compose.chatui.yml up -d

# View logs
docker compose -f docker-compose.yml -f docker-compose.chatui.yml logs -f

# View ChatUI logs only
docker compose -f docker-compose.yml -f docker-compose.chatui.yml logs -f chatui

# Rebuild ChatUI (if config changed)
docker compose -f docker-compose.yml -f docker-compose.chatui.yml build chatui
docker compose -f docker-compose.yml -f docker-compose.chatui.yml up -d chatui

# Stop all services
docker compose -f docker-compose.yml -f docker-compose.chatui.yml down

# Stop and reset database (WARNING: deletes conversations)
docker compose -f docker-compose.yml -f docker-compose.chatui.yml down -v
```

### Accessing Services

- **Chat UI:** http://localhost:3000
- **Model Service:** http://localhost:8000
- **MinIO Console:** http://localhost:9001

---

## Development Stack

**Use Case:** Basic single-service development and testing

**Services Included:**
- MinIO (port 9000) - S3-compatible storage
- Model Service (port 8000) - Single LLM service

### Commands

```bash
# Start basic stack
docker compose up -d

# View logs
docker compose logs -f

# Rebuild after code changes
docker compose build
docker compose up -d

# Stop services
docker compose down
```

---

## Common Operations

### Starting Multiple Stacks

You can run different stacks simultaneously (if ports don't conflict):

```bash
# Option 1: GPU cluster for production + ChatUI for user access
docker compose -f docker-compose.yml -f docker-compose.gpu-cluster.yml up -d
docker compose -f docker-compose.yml -f docker-compose.chatui.yml up -d

# Option 2: Just MinIO + specific services as needed
docker compose up -d minio
docker compose -f docker-compose.gpu-cluster.yml up -d gateway registry
```

### Rebuilding After Code Changes

```bash
# Rebuild specific service (faster)
docker compose -f docker-compose.yml -f docker-compose.gpu-cluster.yml build gateway
docker compose -f docker-compose.yml -f docker-compose.gpu-cluster.yml up -d gateway

# Rebuild all services (thorough)
docker compose -f docker-compose.yml -f docker-compose.gpu-cluster.yml build
docker compose -f docker-compose.yml -f docker-compose.gpu-cluster.yml up -d
```

### Viewing Container Status

```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Check resource usage
docker stats

# Inspect specific container
docker inspect model-gateway
```

### Managing Logs

```bash
# Follow logs in real-time
docker compose logs -f

# View last 50 lines
docker compose logs --tail=50

# View logs from specific time
docker compose logs --since 10m

# View logs for specific service
docker logs model-gateway
docker logs model-registry
```

### Cleaning Up

```bash
# Remove stopped containers
docker compose down

# Remove containers and volumes (deletes data!)
docker compose down -v

# Remove containers, volumes, and orphans
docker compose down -v --remove-orphans

# Remove unused images (free up space)
docker image prune -a

# Nuclear option: remove everything
docker system prune -a --volumes
```

---

## Troubleshooting

### Port Already in Use

If you see "port is already allocated":

```bash
# Find what's using the port
sudo lsof -i :8080
sudo lsof -i :3000

# Stop conflicting service
docker compose down

# Or kill the process
kill -9 <PID>
```

### Container Won't Start

```bash
# Check logs for errors
docker logs <container-name> 2>&1 | tail -20

# Rebuild from scratch
docker compose build --no-cache <service-name>
docker compose up -d <service-name>
```

### Module Not Found Errors

If you see `ModuleNotFoundError`:

```bash
# Rebuild the image (don't use cache)
docker compose build --no-cache <service-name>
docker compose up -d <service-name>
```

### MinIO Connection Issues

```bash
# Check if MinIO is running
docker ps | grep minio

# Restart MinIO
docker compose restart minio

# Check MinIO logs
docker compose logs minio
```

### GPU Not Detected

If services can't access GPU:

```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi

# Check docker-compose.gpu-cluster.yml has correct device_ids
# Ensure NVIDIA Container Toolkit is installed
```

### Network Issues

If services can't communicate:

```bash
# Inspect network
docker network inspect model-service_nginx

# Recreate network
docker compose down
docker compose up -d
```

### Orphaned Containers Warning

If you see "Found orphan containers" warning:

```bash
# Clean up orphans
docker compose down --remove-orphans

# Or ignore the warning (it's just informational)
```

---

## Quick Reference

| Stack | Command | Description |
|-------|---------|-------------|
| GPU Cluster | `docker compose -f docker-compose.yml -f docker-compose.gpu-cluster.yml up -d` | Start production GPU services |
| GPU Cluster | `docker compose -f docker-compose.yml -f docker-compose.gpu-cluster.yml down` | Stop GPU services |
| ChatUI | `docker compose -f docker-compose.yml -f docker-compose.chatui.yml up -d` | Start chat interface |
| ChatUI | `docker compose -f docker-compose.yml -f docker-compose.chatui.yml down` | Stop chat interface |
| Dev | `docker compose up -d` | Start basic dev stack |
| Dev | `docker compose down` | Stop dev stack |
| Logs | `docker compose logs -f <service>` | Follow service logs |
| Rebuild | `docker compose build <service>` | Rebuild specific service |
| Status | `docker ps` | List running containers |
| Clean | `docker compose down -v` | Remove all containers and volumes |

---

## Environment Variables

### Required for All Stacks

Create a `.env` file in the project root:

```env
# MinIO Configuration
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_BUCKET=models

# Model Service
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
MODEL_TYPE=llm
HF_HOME=/models
```

### Required for ChatUI

Create a `chatui.env` file:

```env
MONGODB_URL=mongodb://chatui-mongo:27017
PUBLIC_APP_NAME=My AI Hub
PUBLIC_APP_DESCRIPTION=Offline AI Model Hub
MODELS=[{"name":"My Model","endpoints":[{"type":"openai","baseURL":"http://model-service:8000/v1"}]}]
```

See `chatui.env.template` for full configuration options.

---

## Next Steps

- [GPU Cluster Architecture](./GPU-CLUSTER-GUIDE.md)
- [Alternatives Comparison](./ALTERNATIVES-COMPARISON.md)
- [Main README](../README.md)
