# DotsOCR Docker Container Setup

This directory contains Docker configuration files to containerize the complete DotsOCR API server setup.

## Prerequisites

- Docker Engine 20.10+ with Docker Compose
- NVIDIA Docker runtime (`nvidia-docker2`)
- CUDA-capable GPU with 8GB+ VRAM
- Host system with NVIDIA drivers 450.80.02+

## Quick Start

### 1. Basic Setup

```bash
# Build and start the API server
cd container
docker-compose up --build

# The API will be available at http://localhost:3803
```

### 2. Development Mode

```bash
# Start with development overrides (hot reloading, monitoring)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# This enables:
# - Source code hot reloading
# - GPU monitoring container
# - Redis caching
# - Debug logging
```

### 3. Production Mode

```bash
# Start with production optimizations
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --build

# This enables:
# - Multiple workers
# - Load balancer (nginx)
# - Enhanced health checks
# - Resource limits
# - Redis caching
```

## Container Structure

### Main Service: `dotsocr-api`
- **Base Image**: `nvidia/cuda:12.9.0-devel-ubuntu22.04`
- **Python**: 3.11 with CUDA support
- **Port**: 3803
- **GPU**: Requires NVIDIA GPU with CUDA 12.4+

### Optional Services

#### GPU Monitor (`gpu-monitor`)
- Continuously monitors GPU usage
- Enable with: `docker-compose --profile monitoring up`

#### Redis Cache (`redis`)
- For caching and session management
- Enable with: `docker-compose --profile caching up`

#### Nginx Load Balancer (Production)
- Only available in production mode
- Handles SSL termination and load balancing

## Volume Mounts

```yaml
volumes:
  - ./weights:/app/weights      # Model weights (persistent)
  - ./data:/app/data           # Input data files
  - ./output:/app/output       # OCR results
  - ./config:/app/config       # Configuration files
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device to use |
| `WORKERS` | `1` | Number of API workers |
| `LOG_LEVEL` | `INFO` | Logging level |
| `DEBUG` | `0` | Enable debug mode |
| `HF_HOME` | `/app/cache/huggingface` | HuggingFace cache |

## API Endpoints

Once running, the following endpoints are available:

### Health & Status
- `GET /` - Server status
- `GET /health` - Health check
- `GET /v1/models` - Available models

### Parser Management
- `POST /v1/parser/start` - Load parser model
- `POST /v1/parser/stop` - Unload parser
- `POST /v1/parser/truly-stop` - Force cleanup
- `POST /v1/server/restart` - Restart server

### OCR Processing
- `POST /v1/chat/completions` - OpenAI-compatible OCR
- `POST /v1/ocr/analyze` - Direct OCR analysis

## Memory Management

The container includes GPU memory management features:

```bash
# Check GPU memory usage
docker exec dotsocr-api-server nvidia-smi

# Restart server to free GPU memory
curl -X POST http://localhost:3803/v1/server/restart
```

## Scaling and Load Balancing

### Horizontal Scaling
```bash
# Scale API service to 3 replicas
docker-compose up --scale dotsocr-api=3
```

### Production with Load Balancer
```bash
# Use production setup with nginx
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

## Monitoring

### Container Logs
```bash
# View API server logs
docker-compose logs -f dotsocr-api

# View all service logs
docker-compose logs -f
```

### Resource Usage
```bash
# Monitor container resources
docker stats dotsocr-api-server

# GPU monitoring (if enabled)
docker-compose logs gpu-monitor
```

## Troubleshooting

### Common Issues

#### 1. GPU Not Detected
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.9.0-runtime-ubuntu22.04 nvidia-smi
```

#### 2. Out of GPU Memory
```bash
# Restart container to free memory
docker-compose restart dotsocr-api

# Or use memory management API
curl -X POST http://localhost:3803/v1/server/restart
```

#### 3. Model Download Failed
```bash
# Manually download model
docker exec dotsocr-api-server python3 tools/download_model.py
```

#### 4. Permission Issues
```bash
# Fix volume permissions
sudo chown -R $USER:$USER ./weights ./data ./output
```

### Debugging

#### Access Container Shell
```bash
# Enter running container
docker exec -it dotsocr-api-server bash

# Check Python environment
docker exec dotsocr-api-server python3 -c "import torch; print(torch.cuda.is_available())"
```

#### Development Debug Mode
```bash
# Start with debug enabled
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

## Configuration

### Custom Configuration
Create a `config` directory with custom settings:

```bash
mkdir -p config
echo '{"model_path": "/app/weights/DotsOCR", "max_workers": 2}' > config/api_config.json
```

### SSL Certificate (Production)
```bash
# Place SSL certificates in ssl directory
mkdir -p ssl
cp your-cert.pem ssl/
cp your-key.pem ssl/
```

## Performance Tips

1. **GPU Memory**: Use `--shm-size=8g` for large documents
2. **CPU Cores**: Set `--cpus="4.0"` to limit CPU usage
3. **Memory Limit**: Use `--memory="16g"` to prevent OOM
4. **Caching**: Enable Redis for better performance

## Security Considerations

1. **Network**: Use custom networks in production
2. **Secrets**: Use Docker secrets for API keys
3. **User**: Run as non-root user (add to Dockerfile)
4. **Firewall**: Restrict port access in production

## Example Usage

```bash
# Test the containerized API
curl -X POST http://localhost:3803/v1/parser/start

# Process an image
curl -X POST http://localhost:3803/v1/ocr/analyze \
  -F "image=@/path/to/document.jpg" \
  -F "options={\"format\": \"json\"}"

# Check memory usage
curl http://localhost:3803/v1/parser/status
```

## Maintenance

### Updates
```bash
# Update to latest version
docker-compose pull
docker-compose up --build --force-recreate
```

### Cleanup
```bash
# Remove containers and volumes
docker-compose down -v

# Remove images
docker image prune -f
```

## Support

For issues and questions:
1. Check container logs: `docker-compose logs`
2. Verify GPU access: `docker exec container_name nvidia-smi`
3. Review API documentation in main README.md