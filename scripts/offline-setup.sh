#!/bin/bash
# Offline Model Setup Script
#
# Run this script ONCE when you have internet connectivity to pre-download
# all models to MinIO for offline/air-gapped operation.
#
# Usage: ./scripts/offline-setup.sh

set -e

echo "=========================================="
echo "Model Service - Offline Setup"
echo "=========================================="
echo ""
echo "This script will download models from Hugging Face"
echo "and store them in MinIO for offline use."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if docker-compose is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Start MinIO and Registry services only
echo -e "${BLUE}Starting MinIO and Registry services...${NC}"
docker compose up -d minio registry

# Wait for services to be ready
echo -e "${BLUE}Waiting for services to initialize...${NC}"
sleep 10

# Check if registry is healthy
if ! curl -s http://localhost:8090/health > /dev/null; then
    echo -e "${YELLOW}Warning: Registry service not responding. Waiting longer...${NC}"
    sleep 10
fi

echo ""
echo -e "${GREEN}✓ Services ready${NC}"
echo ""

# Function to download a model
download_model() {
    local model_name=$1
    local hf_repo=$2
    
    echo -e "${BLUE}Downloading ${model_name} (${hf_repo})...${NC}"
    
    response=$(curl -s -X POST "http://localhost:8090/models/${model_name}/download" \
        -H "Content-Type: application/json" \
        -d "{\"hf_repo\": \"${hf_repo}\"}")
    
    if echo "$response" | grep -q "downloading"; then
        echo -e "${GREEN}✓ ${model_name} download started${NC}"
    else
        echo -e "${YELLOW}⚠ ${model_name} may have failed: ${response}${NC}"
    fi
}

# Download recommended models for GPU cluster
echo "Downloading recommended models..."
echo "This may take 10-30 minutes depending on your internet speed."
echo ""

# Primary LLM (small, fast)
download_model "qwen" "Qwen/Qwen2.5-0.5B-Instruct"
sleep 2

# Secondary LLM (alternative)
download_model "tinyllama" "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
sleep 2

# Embedding model
download_model "embedding" "sentence-transformers/all-MiniLM-L6-v2"
sleep 2

# Multimodal (vision) - larger model
echo -e "${YELLOW}Note: LLaVA is ~13GB and will take longer...${NC}"
download_model "llava" "llava-hf/llava-v1.6-mistral-7b-hf"
sleep 2

# Optional: Transcription
# Uncomment if you need speech-to-text
# download_model "whisper" "openai/whisper-base"

echo ""
echo "=========================================="
echo "Download jobs submitted!"
echo "=========================================="
echo ""
echo "Monitor progress with:"
echo "  docker compose logs -f registry"
echo ""
echo "Check downloaded models with:"
echo "  curl http://localhost:8090/models"
echo ""
echo "Once downloads complete, you can run offline with:"
echo "  docker compose -f docker-compose.gpu-cluster.yml up -d"
echo ""
echo -e "${GREEN}Setup complete!${NC}"
