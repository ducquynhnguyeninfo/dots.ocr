# DotsOCR Container Setup Summary

## 📁 Container Directory Structure

```
container/
├── Dockerfile                 # Main container image definition
├── docker-compose.yml         # Base Docker Compose configuration
├── docker-compose.dev.yml     # Development overrides
├── docker-compose.prod.yml    # Production overrides
├── nginx.conf                 # Nginx load balancer configuration
├── run.sh                     # Management script
├── .dockerignore              # Docker build exclusions
└── README.md                  # Comprehensive documentation
```

## 🚀 Quick Start Commands

### 1. Check Requirements
```bash
cd container
./run.sh check
```

### 2. Build and Start (Basic)
```bash
./run.sh build
./run.sh start
```

### 3. Development Mode (with monitoring)
```bash
./run.sh start dev
```

### 4. Production Mode (with load balancer)
```bash
./run.sh start prod
```

## 🔧 Key Features

### Multi-Environment Support
- **Basic**: Single container with API server
- **Development**: Hot reloading + GPU monitoring + Redis
- **Production**: Load balancer + scaling + optimization

### GPU Memory Management
- Full CUDA 12.1 support
- Automatic GPU memory monitoring
- Memory cleanup endpoints: `/v1/server/restart`

### Volume Mounts
- `./weights` → Model storage (persistent)
- `./data` → Input files
- `./output` → OCR results
- `./config` → Configuration files

### Health Monitoring
- Container health checks
- GPU memory monitoring
- API endpoint testing

## 📊 Container Specifications

### Base Image
- `nvidia/cuda:12.9.0-devel-ubuntu22.04`
- Python 3.11 with CUDA support
- PyTorch with CUDA 12.4+ (compatible with CUDA 12.9)

### Resource Requirements
- **Minimum**: 8GB GPU memory, 8GB RAM
- **Recommended**: 16GB GPU memory, 16GB RAM
- **Storage**: 10GB for model weights

### Ports
- **3803**: DotsOCR API server
- **80/443**: Nginx load balancer (production)
- **6379**: Redis cache (optional)

## 🛠️ Management Commands

```bash
# System check
./run.sh check

# Build image
./run.sh build

# Start services
./run.sh start [basic|dev|prod]

# View logs
./run.sh logs [service_name]

# Check status
./run.sh status

# Test API
./run.sh test

# Cleanup
./run.sh cleanup
```

## 🔍 Testing the Setup

### 1. Basic Health Check
```bash
curl http://localhost:3803/
```

### 2. Load Parser
```bash
curl -X POST http://localhost:3803/v1/parser/start
```

### 3. Process Document
```bash
curl -X POST http://localhost:3803/v1/ocr/analyze \
  -F "image=@document.jpg"
```

### 4. Check GPU Memory
```bash
docker exec dotsocr-api-server nvidia-smi
```

## 🎯 Use Cases

### Development
- Code changes reflected instantly
- GPU monitoring enabled
- Debug logging active
- Redis caching available

### Production
- Multiple worker processes
- Nginx load balancing
- SSL termination ready
- Resource limits enforced
- Health checks optimized

### Scaling
- Horizontal scaling: `docker-compose up --scale dotsocr-api=3`
- Load balancing via nginx
- Redis session sharing

## 🔒 Security Features

- Non-root user execution
- Resource limits
- Rate limiting via nginx
- Security headers
- Network isolation
- Volume mount restrictions

## 📈 Performance Optimizations

- Multi-stage Docker build
- Layer caching optimization
- GPU memory management
- Connection pooling
- Static file serving
- Gzip compression

## 🆘 Troubleshooting

### Common Issues
1. **GPU not detected**: Check nvidia-docker2 installation
2. **Out of memory**: Use memory management endpoints
3. **Port conflicts**: Modify port mappings in compose files
4. **Permission issues**: Fix volume mount permissions

### Debug Commands
```bash
# Container shell access
docker exec -it dotsocr-api-server bash

# View all logs
./run.sh logs

# Check GPU status
./run.sh status

# Test API connectivity
./run.sh test
```

## 🔄 Update Process

```bash
# Pull latest changes
git pull origin master

# Rebuild with latest code
./run.sh build

# Restart with new image
./run.sh restart [mode]
```

## 📝 Configuration Files

### Environment Variables
Set in `docker-compose.yml` or `.env` file:
- `CUDA_VISIBLE_DEVICES=0`
- `WORKERS=1`
- `LOG_LEVEL=INFO`

### Custom Configuration
Place in `config/` directory:
- `api_config.json`
- `model_config.yaml`
- Custom environment files

This containerized setup provides a complete, production-ready DotsOCR deployment with comprehensive GPU memory management, monitoring, and scaling capabilities! 🎉