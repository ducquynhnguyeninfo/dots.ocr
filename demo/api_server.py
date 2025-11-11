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

def initialize_parser():
    """Initialize the DotsOCR parser with HF model (thread-safe)"""
    global parser
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
                print("✅ DotsOCR parser initialized successfully!")
    return parser

def process_ocr_task(image: Image.Image, prompt_mode: str, request_id: str):
    """Process OCR task in thread pool (GPU inference is still serialized)"""
    # This function runs in thread pool for CPU-intensive parts
    # GPU inference itself is still serialized by the HF model
    
    ocr_parser = initialize_parser()
    
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

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
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
            request_id
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
        
        # Expect format: {"image": "base64_string", "prompt_mode": "prompt_layout_all_en"}
        if "image" not in data:
            raise HTTPException(status_code=400, detail="Missing 'image' field")
        
        image = decode_base64_image(data["image"])
        prompt_mode = data.get("prompt_mode", "prompt_layout_all_en")
        
        request_id = str(uuid.uuid4())
        print(f"🔍 Processing OCR analysis {request_id} with mode: {prompt_mode}")
        
        # Run OCR processing in thread pool
        loop = asyncio.get_event_loop()
        response_data = await loop.run_in_executor(
            thread_pool, 
            process_ocr_task, 
            image, 
            prompt_mode, 
            request_id
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
    print(f"� GPU serialization: Handled by parser_lock (thread-safe)")
    print(f"🌐 Concurrency limits: Disabled (prevents 503 errors)")
    print(f"📚 OpenAI-compatible endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"🔍 Custom OCR endpoint: http://{args.host}:{args.port}/v1/ocr/analyze")
    print(f"📖 API docs: http://{args.host}:{args.port}/docs")
    
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