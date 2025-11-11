#!/usr/bin/env python3
"""
OpenAI-Compatible API Server for DotsOCR
Provides REST API endpoints compatible with OpenAI chat completions format
"""

import os
import sys
import base64
import json
import time
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import threading

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils.prompts import dict_promptmode_to_prompt


# ==================== Pydantic Models ====================

class ChatMessage(BaseModel):
    role: str
    content: List[Dict[str, Any]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 16384
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


# ==================== FastAPI App ====================

app = FastAPI(
    title="DotsOCR API Server",
    description="OpenAI-compatible API server for DotsOCR document analysis",
    version="1.0.0"
)

# Enable CORS for web applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global parser instance and thread safety
parser = None
parser_lock = threading.Lock()
thread_pool = ThreadPoolExecutor(max_workers=4)  # For CPU-intensive pre/post processing
parser_initialized_at = None  # Track when parser was initialized

def initialize_parser():
    """Initialize the DotsOCR parser with HF model (thread-safe)"""
    global parser, parser_initialized_at
    if parser is None:
        with parser_lock:
            if parser is None:  # Double-check pattern
                print("🚀 Initializing DotsOCR parser with HuggingFace model...")
                parser = DotsOCRParser(
                    use_hf=True,  # Use local HF model instead of vLLM server
                    num_thread=1,  # HF model uses single thread for GPU inference
                    output_dir="./temp_output",
                    min_pixels=None,
                    max_pixels=None,
                )
                parser_initialized_at = time.time()
                print("✅ DotsOCR parser initialized successfully!")
    return parser

def destroy_parser():
    """Destroy the parser and free GPU memory (thread-safe)"""
    global parser, parser_initialized_at
    with parser_lock:
        if parser is not None:
            print("🗑️ Destroying DotsOCR parser and freeing GPU memory...")
            
            # Get initial GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    initial_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    initial_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                    print(f"📊 Initial GPU memory - Allocated: {initial_allocated:.2f} GB, Reserved: {initial_reserved:.2f} GB")
            except:
                initial_allocated = initial_reserved = None
            
            # Clear model from GPU memory
            if hasattr(parser, 'model') and parser.model is not None:
                try:
                    print("🔄 Moving model to CPU...")
                    # Move model to CPU first
                    if hasattr(parser.model, 'cpu'):
                        parser.model.cpu()
                    
                    # Delete model components
                    if hasattr(parser, 'processor') and parser.processor is not None:
                        del parser.processor
                        parser.processor = None
                    
                    # Delete the model itself
                    del parser.model
                    parser.model = None
                    
                    print("🧹 Model components deleted")
                except Exception as e:
                    print(f"⚠️ Warning during model cleanup: {e}")
            
            # Clear any other parser attributes that might hold GPU tensors
            try:
                if hasattr(parser, 'process_vision_info'):
                    parser.process_vision_info = None
                print("🗑️ Cleared additional parser attributes")
            except Exception as e:
                print(f"⚠️ Warning during attribute cleanup: {e}")
            
            # Clear the parser itself
            parser = None
            parser_initialized_at = None
            
            # AGGRESSIVE GPU memory cleanup
            try:
                import gc
                import torch
                
                print("🔄 Running multiple garbage collection passes...")
                for i in range(3):  # Multiple GC passes
                    gc.collect()
                    time.sleep(0.1)  # Small delay between passes
                
                if torch.cuda.is_available():
                    print("🔄 Aggressive GPU memory cleanup...")
                    
                    # Clear all GPU caches multiple times
                    for i in range(3):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        time.sleep(0.1)
                    
                    # Clear IPC cache
                    torch.cuda.ipc_collect()
                    
                    # Force reset memory stats (helps with fragmentation)
                    try:
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.reset_accumulated_memory_stats()
                    except:
                        pass
                    
                    # Final cleanup
                    torch.cuda.empty_cache()
                    
                    # Get final GPU memory
                    final_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    final_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                    print(f"📊 Final GPU memory - Allocated: {final_allocated:.2f} GB, Reserved: {final_reserved:.2f} GB")
                    
                    if initial_allocated is not None and initial_reserved is not None:
                        freed_allocated = initial_allocated - final_allocated
                        freed_reserved = initial_reserved - final_reserved
                        print(f"✅ Freed - Allocated: {freed_allocated:.2f} GB, Reserved: {freed_reserved:.2f} GB")
                        
                        if final_reserved > 1.0:  # Still more than 1GB reserved
                            print(f"⚠️ WARNING: {final_reserved:.2f} GB still reserved by PyTorch")
                            print("   This is normal PyTorch behavior - it caches GPU memory for performance")
                            print("   To fully free GPU memory, restart the entire Python process")
                
                print("🧹 Aggressive cleanup completed")
            except Exception as e:
                print(f"⚠️ Warning during final cleanup: {e}")
            
            print("✅ DotsOCR parser destroyed successfully!")
            return True
        else:
            print("ℹ️ Parser was not initialized, nothing to destroy")
            return False

def force_gpu_memory_reset():
    """Force complete GPU memory reset - WARNING: This may affect other GPU processes"""
    try:
        import torch
        if torch.cuda.is_available():
            print("🚨 FORCING COMPLETE GPU MEMORY RESET...")
            
            # Clear all caches
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            # Reset memory management
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            
            # Try to reset cached allocator (this is the nuclear option)
            try:
                # This will free ALL cached memory, potentially affecting other processes
                torch.cuda.empty_cache()
                print("✅ Complete GPU memory reset attempted")
                
                # Show final memory state
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"📊 After reset - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
                
                return True
            except Exception as e:
                print(f"⚠️ Nuclear memory reset failed: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Force GPU memory reset failed: {e}")
        return False

def get_parser_status():
    """Get current parser status and memory info"""
    global parser, parser_initialized_at
    
    # Get GPU memory info
    gpu_info = {"available": False}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "memory_cached_gb": torch.cuda.memory_cached() / 1024**3,
                "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
    except Exception as e:
        gpu_info["error"] = str(e)
    
    status = {
        "parser_loaded": parser is not None,
        "initialized_at": parser_initialized_at,
        "uptime_seconds": time.time() - parser_initialized_at if parser_initialized_at else 0,
        "gpu_info": {}
    }
    
    # Get GPU memory info if available
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            status["gpu_info"] = {
                "available": True,
                "device_count": device_count,
                "devices": []
            }
            
            for i in range(device_count):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                
                status["gpu_info"]["devices"].append({
                    "device_id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated_gb": round(memory_allocated, 2),
                    "memory_reserved_gb": round(memory_reserved, 2),
                    "memory_total_gb": round(memory_total, 2),
                    "memory_free_gb": round(memory_total - memory_reserved, 2)
                })
        else:
            status["gpu_info"] = {"available": False, "reason": "CUDA not available"}
    except Exception as e:
        status["gpu_info"] = {"available": False, "error": str(e)}
    
    return status

def process_ocr_task(image: Image.Image, prompt_mode: str, request_id: str, auto_start: bool = True):
    """Process OCR task in thread pool (GPU inference is still serialized)"""
    # This function runs in thread pool for CPU-intensive parts
    # GPU inference itself is still serialized by the HF model
    
    global parser
    
    # Check if parser is initialized
    if parser is None:
        if auto_start:
            print(f"⚡ Parser not initialized, auto-starting for request {request_id}")
            ocr_parser = initialize_parser()
        else:
            raise Exception("Parser not initialized. Please start the parser first using POST /v1/parser/start")
    else:
        ocr_parser = parser
    
    # Pre-processing can be done in parallel
    temp_dir = f"./temp_output/{request_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # The actual GPU inference will be serialized by the model
        # But file I/O and pre/post processing can benefit from threading
        with parser_lock:  # Ensure GPU model access is serialized
            results = ocr_parser.parse_image(
                input_path=image,
                filename=f"api_request_{request_id}",
                prompt_mode=prompt_mode,
                save_dir=temp_dir,
                fitz_preprocess=False
            )
        
        if not results:
            raise Exception("No results from OCR processing")
        
        result = results[0]
        
        # Post-processing can be done in parallel
        response_data = {
            "request_id": request_id,
            "status": "success",
            "input_width": result.get("input_width", 0),
            "input_height": result.get("input_height", 0),
            "page_no": result.get("page_no", 0),
            "markdown_content": None,
            "layout_data": None,
            "filtered": result.get("filtered", False)
        }
        
        # File I/O can benefit from being in thread pool
        if 'md_content_path' in result and os.path.exists(result['md_content_path']):
            with open(result['md_content_path'], 'r', encoding='utf-8') as f:
                response_data["markdown_content"] = f.read()
        
        if 'layout_info_path' in result and os.path.exists(result['layout_info_path']):
            with open(result['layout_info_path'], 'r', encoding='utf-8') as f:
                response_data["layout_data"] = json.load(f)
        
        return response_data
        
    finally:
        # Cleanup
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 image string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def extract_image_and_prompt(messages: List[ChatMessage]) -> tuple[Image.Image, str]:
    """Extract image and text prompt from OpenAI chat messages format"""
    image = None
    text_parts = []
    
    for message in messages:
        if message.role == "user":
            for content in message.content:
                if content["type"] == "image_url":
                    image_url = content["image_url"]["url"]
                    image = decode_base64_image(image_url)
                elif content["type"] == "text":
                    text_parts.append(content["text"])
    
    if image is None:
        raise HTTPException(status_code=400, detail="No image found in request")
    
    # Use default prompt if no text provided
    prompt_text = " ".join(text_parts) if text_parts else "prompt_layout_all_en"
    
    # Map common prompt names to actual prompts
    if prompt_text in dict_promptmode_to_prompt:
        prompt_text = dict_promptmode_to_prompt[prompt_text]
    
    return image, prompt_text


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "DotsOCR API Server is running"}

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "dots-ocr",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "rednote-hilab"
            }
        ]
    }

# ==================== Parser Management Endpoints ====================

@app.get("/v1/parser/status")
async def parser_status():
    """Get current parser status and GPU memory usage"""
    try:
        status = get_parser_status()
        return {
            "status": "success",
            "data": status
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/v1/parser/start")
async def start_parser():
    """Initialize the parser and load model into GPU memory"""
    try:
        global parser
        if parser is not None:
            return {
                "status": "info",
                "message": "Parser is already initialized",
                "data": get_parser_status()
            }
        
        # Initialize parser in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(thread_pool, initialize_parser)
        
        return {
            "status": "success", 
            "message": "Parser initialized successfully",
            "data": get_parser_status()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to initialize parser: {str(e)}",
            "error": str(e)
        }

@app.post("/v1/parser/stop")
async def stop_parser():
    """Destroy the parser and free GPU memory"""
    try:
        # Destroy parser in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        destroyed = await loop.run_in_executor(thread_pool, destroy_parser)
        
        if destroyed:
            return {
                "status": "success",
                "message": "Parser destroyed and GPU memory freed",
                "data": get_parser_status()
            }
        else:
            return {
                "status": "info",
                "message": "Parser was not initialized, nothing to destroy",
                "data": get_parser_status()
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to destroy parser: {str(e)}",
            "error": str(e)
        }

@app.post("/v1/parser/restart")
async def restart_parser():
    """Restart the parser (destroy and reinitialize)"""
    try:
        # Stop parser first
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(thread_pool, destroy_parser)
        
        # Small delay to ensure cleanup
        await asyncio.sleep(1)
        
        # Start parser again
        await loop.run_in_executor(thread_pool, initialize_parser)
        
        return {
            "status": "success",
            "message": "Parser restarted successfully",
            "data": get_parser_status()
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Failed to restart parser: {str(e)}",
            "error": str(e)
        }

@app.post("/v1/parser/force-reset-memory")
async def force_reset_gpu_memory():
    """NUCLEAR OPTION: Force complete GPU memory reset - may affect other processes"""
    try:
        # First destroy parser if loaded
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(thread_pool, destroy_parser)
        
        # Wait a moment
        await asyncio.sleep(0.5)
        
        # Force memory reset
        reset_success = await loop.run_in_executor(thread_pool, force_gpu_memory_reset)
        
        return {
            "status": "success" if reset_success else "partial",
            "message": "Nuclear GPU memory reset attempted - check GPU memory usage",
            "warning": "This may have affected other GPU processes on the system",
            "data": get_parser_status()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to force GPU memory reset: {str(e)}",
            "error": str(e)
        }

@app.post("/v1/server/restart")
async def restart_server():
    """ULTIMATE SOLUTION: Restart the entire server process to truly free GPU memory"""
    try:
        # Import required modules
        import os
        import sys
        
        # First try to destroy parser gracefully
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(thread_pool, destroy_parser)
        
        # Schedule server restart after response is sent
        async def delayed_restart():
            await asyncio.sleep(1)  # Give time for response to be sent
            print("🔄 RESTARTING SERVER PROCESS TO FREE GPU MEMORY...")
            
            # Get current command line arguments
            python_exe = sys.executable
            script_path = os.path.abspath(__file__)
            cmd_args = sys.argv[1:]  # Get original command line arguments
            
            # Restart the process
            os.execv(python_exe, [python_exe, script_path] + cmd_args)
        
        # Schedule the restart
        asyncio.create_task(delayed_restart())
        
        return {
            "status": "success",
            "message": "Server restart initiated - GPU memory will be completely freed",
            "warning": "Server will be unavailable for a few seconds during restart",
            "restart_in_seconds": 1
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to restart server: {str(e)}",
            "error": str(e)
        }

@app.post("/v1/parser/truly-stop")
async def truly_stop_parser():
    """Enhanced stop that tries multiple cleanup strategies"""
    try:
        import torch
        import gc
        import os
        
        # Get initial GPU memory
        initial_reserved = 0
        if torch.cuda.is_available():
            initial_reserved = torch.cuda.memory_reserved() / 1024**3
        
        print(f"🗑️ TRULY STOPPING PARSER - Initial Reserved: {initial_reserved:.2f} GB")
        
        # Strategy 1: Standard destroy
        loop = asyncio.get_event_loop()
        destroyed = await loop.run_in_executor(thread_pool, destroy_parser)
        
        # Strategy 2: Force Python to release references
        def aggressive_cleanup():
            global parser
            
            # Clear all possible references
            parser = None
            
            # Multiple garbage collection passes
            for i in range(5):
                collected = gc.collect()
                print(f"   GC pass {i+1}: collected {collected} objects")
            
            # Clear GPU memory multiple times
            if torch.cuda.is_available():
                for i in range(5):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Try to reset the allocator (this is the key!)
                try:
                    # This forces PyTorch to release reserved memory back to CUDA driver
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()
                    torch.cuda.empty_cache()
                    
                    # Force CUDA context to be destroyed and recreated
                    torch.cuda.synchronize()
                    print("   ✅ CUDA context reset attempted")
                except Exception as e:
                    print(f"   ⚠️ CUDA context reset failed: {e}")
                
                # Get final memory
                final_reserved = torch.cuda.memory_reserved() / 1024**3
                freed = initial_reserved - final_reserved
                print(f"   📊 Reserved memory freed: {freed:.2f} GB")
                
                return final_reserved < 0.1  # Success if less than 100MB reserved
            
            return True
        
        # Run aggressive cleanup
        cleanup_success = await loop.run_in_executor(thread_pool, aggressive_cleanup)
        
        # Get final status
        final_status = get_parser_status()
        final_reserved = 0
        if final_status.get('gpu_info', {}).get('available'):
            final_reserved = final_status['gpu_info']['devices'][0]['memory_reserved_gb']
        
        message = f"Parser destroyed - Reserved memory: {final_reserved:.2f} GB"
        
        if final_reserved > 1.0:
            message += " (PyTorch still caching memory - use /v1/server/restart for complete cleanup)"
        
        return {
            "status": "success",
            "message": message,
            "memory_truly_freed": cleanup_success,
            "recommendation": "Use /v1/server/restart for guaranteed memory cleanup" if final_reserved > 1.0 else "Memory successfully freed",
            "data": final_status
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to truly stop parser: {str(e)}",
            "error": str(e)
        }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, auto_start: bool = True):
    """OpenAI-compatible chat completions endpoint with async processing"""
    try:
        # Extract image and prompt from request
        image, prompt = extract_image_and_prompt(request.messages)
        
        request_id = str(uuid.uuid4())
        print(f"🔍 Processing request {request_id} with prompt: {prompt[:100]}...")
        
        # Run OCR processing in thread pool (CPU parts can be parallel)
        loop = asyncio.get_event_loop()
        result_data = await loop.run_in_executor(
            thread_pool, 
            process_ocr_task, 
            image, 
            "prompt_layout_all_en",  # Default mode
            request_id,
            auto_start  # Whether to auto-start parser if not initialized
        )
        
        response_content = result_data.get("markdown_content", "OCR processing completed but no markdown content generated.")
        
        # Format response in OpenAI format
        response = ChatCompletionResponse(
            id=f"chatcmpl-{request_id}",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": 0,  # Could implement token counting
                "completion_tokens": 0,
                "total_tokens": 0
            }
        )
        
        print(f"✅ Successfully processed request {request_id}")
        return response
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/v1/ocr/analyze")
async def ocr_analyze(request: Request):
    """Custom endpoint for OCR analysis with more detailed response and async processing"""
    try:
        # Parse JSON request
        data = await request.json()
        
        # Expect format: {"image": "base64_string", "prompt_mode": "prompt_layout_all_en", "auto_start": true}
        if "image" not in data:
            raise HTTPException(status_code=400, detail="Missing 'image' field")
        
        image = decode_base64_image(data["image"])
        prompt_mode = data.get("prompt_mode", "prompt_layout_all_en")
        auto_start = data.get("auto_start", True)  # Default to auto-start
        
        request_id = str(uuid.uuid4())
        print(f"🔍 Processing OCR analysis {request_id} with mode: {prompt_mode}")
        
        # Run OCR processing in thread pool
        loop = asyncio.get_event_loop()
        response_data = await loop.run_in_executor(
            thread_pool, 
            process_ocr_task, 
            image, 
            prompt_mode, 
            request_id,
            auto_start
        )
        
        print(f"✅ Successfully processed OCR analysis {request_id}")
        return response_data
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in OCR analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ==================== Server Startup ====================

if __name__ == "__main__":
    import argparse
    
    parser_args = argparse.ArgumentParser(description="DotsOCR API Server")
    parser_args.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser_args.add_argument("--port", default=8000, type=int, help="Port to bind to")
    parser_args.add_argument("--workers", default=1, type=int, help="Number of worker processes (use 1 for HF model)")
    parser_args.add_argument("--disable-concurrency-limit", action="store_true", 
                       help="Disable uvicorn concurrency limits (recommended for OCR)")
    
    args = parser_args.parse_args()
    
    print(f"🚀 Starting DotsOCR API Server on {args.host}:{args.port}")
    print(f"👥 Workers: {args.workers} (HF model uses single GPU)")
    print(f"🔒 GPU serialization: Handled by parser_lock (thread-safe)")
    print(f"🌐 Concurrency limits: Disabled (prevents 503 errors)")
    print(f"⚡ Parser management: On-demand loading/unloading")
    print()
    print("📚 Available Endpoints:")
    print(f"   • OpenAI Chat: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"   • Custom OCR: http://{args.host}:{args.port}/v1/ocr/analyze")
    print(f"   • Parser Status: http://{args.host}:{args.port}/v1/parser/status")
    print(f"   • Start Parser: http://{args.host}:{args.port}/v1/parser/start")
    print(f"   • Stop Parser: http://{args.host}:{args.port}/v1/parser/stop")
    print(f"   • Truly Stop: http://{args.host}:{args.port}/v1/parser/truly-stop")
    print(f"   • Restart Parser: http://{args.host}:{args.port}/v1/parser/restart")
    print(f"   • Force Memory Reset: http://{args.host}:{args.port}/v1/parser/force-reset-memory")
    print(f"   • Restart Server: http://{args.host}:{args.port}/v1/server/restart")
    print(f"   • API Docs: http://{args.host}:{args.port}/docs")
    print()
    print("🛠️ Management Tool:")
    print(f"   python demo/parser_manager.py --server http://{args.host}:{args.port} <command>")
    
    # For HF model, use single worker but allow async handling
    if args.workers > 1:
        print("⚠️  WARNING: Multiple workers with HF model may cause GPU memory issues!")
        print("   Recommendation: Use workers=1 for HF models")
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False,
        # Remove concurrency limits to avoid 503 errors
        # The GPU serialization is handled by parser_lock instead
        limit_concurrency=None,  # Allow unlimited concurrent connections
        limit_max_requests=None,  # No request limit
        # Increase timeouts for OCR processing
        timeout_keep_alive=30,
        timeout_graceful_shutdown=30
    )