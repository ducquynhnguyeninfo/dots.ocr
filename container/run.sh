#!/bin/bash

# DotsOCR Docker Management Script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check NVIDIA Docker
    if ! docker run --rm --gpus all nvidia/cuda:12.9.0-runtime-ubuntu22.04 nvidia-smi &> /dev/null; then
        log_error "NVIDIA Docker runtime is not properly configured"
        exit 1
    fi
    
    log_success "All requirements met"
}

build() {
    log_info "Building DotsOCR Docker image..."
    docker compose build --no-cache
    log_success "Build completed"
}

start() {
    MODE=${1:-"basic"}
    
    log_info "Starting DotsOCR API server in $MODE mode..."
    
    case $MODE in
        "dev"|"development")
            docker compose -f docker compose.yml -f docker compose.dev.yml up -d
            ;;
        "prod"|"production")
            docker compose -f docker compose.yml -f docker compose.prod.yml up -d
            ;;
        "basic"|*)
            docker compose up -d
            ;;
    esac
    
    log_success "DotsOCR API server started"
    log_info "API available at http://localhost:3803"
    log_info "Use 'docker compose logs -f' to view logs"
}

stop() {
    log_info "Stopping DotsOCR services..."
    docker compose down
    log_success "Services stopped"
}

restart() {
    MODE=${1:-"basic"}
    log_info "Restarting DotsOCR services..."
    stop
    start "$MODE"
}

logs() {
    SERVICE=${1:-""}
    if [ -z "$SERVICE" ]; then
        docker compose logs -f
    else
        docker compose logs -f "$SERVICE"
    fi
}

status() {
    log_info "Service Status:"
    docker compose ps
    
    log_info "GPU Status:"
    if docker ps --format "table {{.Names}}" | grep -q "dotsocr-api-server"; then
        docker exec dotsocr-api-server nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits || log_warning "Could not get GPU status"
    else
        log_warning "DotsOCR container not running"
    fi
}

test_api() {
    log_info "Testing API endpoints..."
    
    # Test health endpoint
    if curl -s http://localhost:3803/ | grep -q "ok"; then
        log_success "Health endpoint: OK"
    else
        log_error "Health endpoint: FAILED"
        return 1
    fi
    
    # Test parser start
    if curl -s -X POST http://localhost:3803/v1/parser/start | grep -q "success"; then
        log_success "Parser start: OK"
    else
        log_warning "Parser start: FAILED (might already be loaded)"
    fi
    
    log_success "API test completed"
}

cleanup() {
    log_info "Cleaning up Docker resources..."
    docker compose down -v
    docker system prune -f
    log_success "Cleanup completed"
}

usage() {
    echo "DotsOCR Docker Management Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  check       Check system requirements"
    echo "  build       Build Docker image"
    echo "  start [MODE] Start services (modes: basic, dev, prod)"
    echo "  stop        Stop services"
    echo "  restart [MODE] Restart services"
    echo "  logs [SERVICE] Show logs"
    echo "  status      Show service and GPU status"
    echo "  test        Test API endpoints"
    echo "  cleanup     Clean up Docker resources"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 check"
    echo "  $0 build"
    echo "  $0 start dev"
    echo "  $0 logs dotsocr-api"
    echo "  $0 test"
}

# Main script logic
case "${1:-help}" in
    "check")
        check_requirements
        ;;
    "build")
        check_requirements
        build
        ;;
    "start")
        check_requirements
        start "$2"
        ;;
    "stop")
        stop
        ;;
    "restart")
        restart "$2"
        ;;
    "logs")
        logs "$2"
        ;;
    "status")
        status
        ;;
    "test")
        test_api
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|*)
        usage
        ;;
esac