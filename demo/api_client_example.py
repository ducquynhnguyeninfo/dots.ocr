#!/usr/bin/env python3
"""
Example client for DotsOCR API Server
Shows how to use both OpenAI-compatible and custom endpoints
"""

import base64
import json
import requests
from PIL import Image
from io import BytesIO

def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def pil_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

class DotsOCRClient:
    def __init__(self, base_url: str = "http://localhost:18000"):
        self.base_url = base_url.rstrip('/')
    
    def chat_completion(self, image_path: str, prompt: str = "Extract text and layout"):
        """Use OpenAI-compatible chat completions endpoint"""
        url = f"{self.base_url}/v1/chat/completions"
        
        # Convert image to base64
        image_b64 = image_to_base64(image_path)
        
        payload = {
            "model": "dots-ocr",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 16384
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def ocr_analyze(self, image_path: str, prompt_mode: str = "prompt_layout_all_en"):
        """Use custom OCR analysis endpoint with detailed response"""
        url = f"{self.base_url}/v1/ocr/analyze"
        
        # Convert image to base64
        image_b64 = image_to_base64(image_path)
        
        payload = {
            "image": image_b64,
            "prompt_mode": prompt_mode
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def health_check(self):
        """Check if the server is running"""
        url = f"{self.base_url}/"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def list_models(self):
        """List available models"""
        url = f"{self.base_url}/v1/models"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

# ==================== Example Usage ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DotsOCR API Client Example")
    parser.add_argument("--server", default="http://localhost:8000", help="API server URL")
    parser.add_argument("--image", required=True, help="Path to image file to process")
    parser.add_argument("--mode", default="prompt_layout_all_en", 
                       choices=["prompt_layout_all_en", "prompt_layout_only_en", "prompt_ocr"],
                       help="Processing mode")
    parser.add_argument("--endpoint", default="chat", choices=["chat", "ocr"],
                       help="Which endpoint to use")
    
    args = parser.parse_args()
    
    # Create client
    client = DotsOCRClient(args.server)
    
    try:
        # Health check
        print("🔍 Checking server health...")
        health = client.health_check()
        print(f"✅ Server status: {health['status']}")
        
        # List models
        print("\n📚 Available models:")
        models = client.list_models()
        for model in models['data']:
            print(f"  - {model['id']}")
        
        print(f"\n🖼️  Processing image: {args.image}")
        print(f"🔧 Using mode: {args.mode}")
        print(f"🌐 Using endpoint: {args.endpoint}")
        
        if args.endpoint == "chat":
            # Use OpenAI-compatible endpoint
            print("\n📡 Using OpenAI-compatible chat completions endpoint...")
            result = client.chat_completion(args.image, args.mode)
            
            print("\n📝 Result:")
            print(f"  Request ID: {result['id']}")
            print(f"  Model: {result['model']}")
            print(f"  Content:\n{result['choices'][0]['message']['content']}")
            
        elif args.endpoint == "ocr":
            # Use custom OCR endpoint
            print("\n📡 Using custom OCR analysis endpoint...")
            result = client.ocr_analyze(args.image, args.mode)
            
            print("\n📝 Detailed Result:")
            print(f"  Request ID: {result['request_id']}")
            print(f"  Status: {result['status']}")
            print(f"  Input Size: {result['input_width']} x {result['input_height']}")
            print(f"  Filtered: {result['filtered']}")
            
            if result['markdown_content']:
                print(f"\n📄 Markdown Content:")
                print(result['markdown_content'])
            
            if result['layout_data']:
                print(f"\n🏗️  Layout Data: {len(result['layout_data'])} elements detected")
                # Optionally save layout data to file
                with open(f"layout_output_{result['request_id']}.json", 'w') as f:
                    json.dump(result['layout_data'], f, indent=2, ensure_ascii=False)
                print(f"  Layout data saved to: layout_output_{result['request_id']}.json")
    
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to server at {args.server}")
        print("   Make sure the API server is running with: python demo/api_server.py")
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP error: {e}")
        print(f"   Response: {e.response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")