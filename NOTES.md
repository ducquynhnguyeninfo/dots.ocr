# DotsOCR Project Notes

## Overview
DotsOCR is a document OCR and layout analysis system that can run in multiple modes: local HuggingFace model or remote vLLM server.

## Architecture & Deployment Modes

### 1. **Two Main Modes**

#### **Mode 1: vLLM Server (`use_hf=False`)**
- **Architecture**: Client-Server
- **Server**: Remote vLLM server (can be local or remote)
- **Client**: DotsOCRParser acts as HTTP client
- **Communication**: OpenAI-compatible API over HTTP/HTTPS
- **GPU Usage**: On server side
- **Threading**: Supports multiple threads (`num_thread=64` default)

```python
ocr = DotsOCRParser(
    ip='localhost',          # Can be any IP/domain
    port=8000,               # Any port
    protocol='https',        # For remote/secure connections  
    use_hf=False            # Use vLLM server mode
)
```

#### **Mode 2: Local HuggingFace (`use_hf=True`)**
- **Architecture**: Local inference
- **Model**: Loaded directly via HuggingFace transformers
- **GPU Usage**: Local GPU required
- **Threading**: Forced to single thread (`num_thread=1`)
- **Reason**: Prevents GPU memory conflicts and ensures thread safety

```python
ocr = DotsOCRParser(
    use_hf=True,            # Use local HF model
    num_thread=1            # Automatically set to 1
)
```

### 2. **vLLM Server Setup**

#### **Starting vLLM Server**
```bash
# Launch vLLM server (from demo/launch_model_vllm.sh)
cd dots.ocr
CUDA_VISIBLE_DEVICES=0 vllm serve ./weights/DotsOCR \
    --host 0.0.0.0 \        # Listen on all interfaces for remote access
    --port 8000 \           # Port number
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code
```

#### **Remote Access Considerations**
- **Security**: Use HTTPS for internet connections
- **Authentication**: Support API keys via environment variable
- **Data Privacy**: Images and text are sent to remote server
- **Network**: Requires stable internet connection

## User Interface Options

### 1. **Gradio Web Interface** (`demo/demo_gradio.py`)
- **Purpose**: Human-friendly web UI
- **Features**: 
  - File upload (PDF/images)
  - Visual result display
  - Interactive page navigation
  - Download results
- **Usage**: `python demo/demo_gradio.py 7860`
- **Access**: Web browser at `http://localhost:7860`

### 2. **OpenAI-Compatible API Server** (`demo/api_server.py`)
- **Purpose**: Machine-to-machine API
- **Protocols**: REST API with JSON
- **Endpoints**:
  - `POST /v1/chat/completions` - OpenAI-compatible
  - `POST /v1/ocr/analyze` - Custom detailed endpoint
  - `GET /v1/models` - List available models

### 3. **API Server Improvements**

#### **Threading & Concurrency**
- **Problem**: HF model can't use multiple GPU threads safely
- **Solution**: 
  - GPU inference serialized with `parser_lock`
  - CPU operations (file I/O, preprocessing) use thread pool
  - Async request handling for better throughput

#### **Concurrency Limit Fix**
- **Issue**: Uvicorn's `limit_concurrency` caused HTTP 503 errors
- **Fix**: Set `limit_concurrency=None` to disable limits
- **Reasoning**: GPU serialization handled by application-level locks

```python
# Fixed configuration
uvicorn.run(
    app,
    limit_concurrency=None,      # No uvicorn limits
    timeout_keep_alive=30        # Longer timeouts for OCR
)
```

## Performance Characteristics

### **vLLM Server Mode**
- ✅ **Scalability**: Multiple clients can connect
- ✅ **Resource Management**: Server handles GPU efficiently  
- ✅ **Throughput**: Optimized for batch processing
- ❌ **Latency**: Network overhead
- ❌ **Dependencies**: Requires separate server setup

### **HuggingFace Local Mode**
- ✅ **Latency**: No network overhead
- ✅ **Privacy**: All processing local
- ✅ **Simplicity**: Single process setup
- ❌ **Scalability**: Limited to single GPU thread
- ❌ **Memory**: Model loaded in each process

## Threading Deep Dive

### **Why HF Model Uses Single Thread**
1. **GPU Memory**: Multiple threads loading same model → memory conflicts
2. **Thread Safety**: HF models not designed for concurrent inference
3. **Model State**: Shared model state can cause race conditions

### **Multi-threading Strategy for HF Mode**
```python
# What we implemented:
- GPU inference: Serialized with parser_lock
- File I/O: Parallel in ThreadPoolExecutor  
- Request handling: Async/await for concurrency
- Pre/post processing: Can run in parallel
```

## API Usage Examples

### **OpenAI SDK Compatible**
```python
import openai

client = openai.OpenAI(
    api_key="dummy",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="dots-ocr",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            {"type": "text", "text": "Extract layout and text"}
        ]
    }]
)
```

### **Custom Client**
```python
# Use provided client
python demo/api_client_example.py --image path/to/image.jpg --server http://localhost:8000
```

## Troubleshooting

### **Common Issues**

#### **1. SSH Configuration Error**
```
/home/user/.ssh/config: line 13: Bad configuration option: usekeychain
```
**Fix**: Remove `UseKeychain` option (macOS-specific, not Linux)
```bash
sed -i '/UseKeychain/d' ~/.ssh/config
```

#### **2. HTTP 503 Service Unavailable**
```
WARNING: Exceeded concurrency limit.
```
**Fix**: Disable uvicorn concurrency limits in API server

#### **3. Missing Dependencies**
```
ModuleNotFoundError: No module named 'tqdm'
```
**Fix**: Install missing packages
```bash
pip install tqdm fastapi uvicorn
```

#### **4. GPU Memory Issues**
- Use `use_hf=True` with `workers=1`
- Don't run multiple HF instances simultaneously
- Monitor GPU memory with `nvidia-smi`

### **Port Management**
- **Default Ports**:
  - vLLM server: 8000
  - Gradio UI: 7860
  - API server: 8000 (configurable)
- **Check port availability**: `python -c "import socket; s=socket.socket(); s.bind(('0.0.0.0', PORT)); print('Available')"`

## File Structure & Key Components

```
dots.ocr/
├── demo/
│   ├── api_server.py          # OpenAI-compatible API server
│   ├── api_client_example.py  # Example API client
│   ├── demo_gradio.py         # Web UI interface
│   ├── launch_model_vllm.sh   # vLLM server launcher
│   └── demo_vllm.py          # vLLM client example
├── dots_ocr/
│   ├── parser.py             # Main DotsOCRParser class
│   ├── model/inference.py    # vLLM inference functions
│   └── utils/               # Utilities and prompts
└── weights/DotsOCR/         # Model weights
```

## Configuration Examples

### **Development Setup**
```bash
# Local development with HF model
python demo/api_server.py --host 127.0.0.1 --port 8000 --workers 1
```

### **Production Setup**
```bash
# Production with vLLM server
# Terminal 1: Start vLLM server
bash demo/launch_model_vllm.sh

# Terminal 2: Start API server (vLLM mode)
python demo/api_server.py --host 0.0.0.0 --port 80 --workers 4
```

### **Remote Server Setup**
```python
# Connect to remote vLLM server
ocr = DotsOCRParser(
    protocol='https',
    ip='your-server.com',
    port=443,
    use_hf=False
)
```

## Git Repository
- **Original**: `https://github.com/rednote-hilab/dots.ocr.git`
- **Fork**: `git@github.com:ducquynhnguyeninfo/dots.ocr.git`
- **Authentication**: SSH keys required for push access

---

*Last updated: November 12, 2025*