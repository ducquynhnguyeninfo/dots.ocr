#!/usr/bin/env python3
"""
DotsOCR Parser Management Client
Provides easy command-line interface to manage parser lifecycle
"""

import argparse
import requests
import json
import time
from typing import Dict, Any

class ParserManagerClient:
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url.rstrip('/')
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[Any, Any]:
        """Make HTTP request to server"""
        url = f"{self.server_url}{endpoint}"
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "error": str(e)}
    
    def get_status(self) -> Dict[Any, Any]:
        """Get parser status and GPU memory info"""
        return self._make_request("GET", "/v1/parser/status")
    
    def start_parser(self) -> Dict[Any, Any]:
        """Start/initialize the parser"""
        return self._make_request("POST", "/v1/parser/start")
    
    def stop_parser(self) -> Dict[Any, Any]:
        """Stop/destroy the parser"""
        return self._make_request("POST", "/v1/parser/stop")
    
    def restart_parser(self) -> Dict[Any, Any]:
        """Restart the parser"""
        return self._make_request("POST", "/v1/parser/restart")
    
    def health_check(self) -> Dict[Any, Any]:
        """Check if server is running"""
        return self._make_request("GET", "/")

def format_status(status_data: Dict[Any, Any]) -> str:
    """Format status data for display"""
    if status_data.get("status") != "success":
        return f"❌ Error: {status_data.get('error', 'Unknown error')}"
    
    data = status_data.get("data", {})
    parser_loaded = data.get("parser_loaded", False)
    uptime = data.get("uptime_seconds", 0)
    gpu_info = data.get("gpu_info", {})
    
    lines = []
    lines.append(f"🤖 Parser Status: {'✅ LOADED' if parser_loaded else '❌ NOT LOADED'}")
    
    if parser_loaded:
        lines.append(f"⏰ Uptime: {uptime:.1f} seconds ({uptime/60:.1f} minutes)")
        init_time = data.get("initialized_at")
        if init_time:
            init_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(init_time))
            lines.append(f"🚀 Initialized: {init_str}")
    
    # GPU Information
    if gpu_info.get("available"):
        lines.append(f"\n🎮 GPU Information:")
        lines.append(f"   Devices: {gpu_info.get('device_count', 0)}")
        
        for device in gpu_info.get("devices", []):
            name = device.get("name", "Unknown")
            allocated = device.get("memory_allocated_gb", 0)
            total = device.get("memory_total_gb", 0)
            free = device.get("memory_free_gb", 0)
            
            lines.append(f"   Device {device.get('device_id', 0)}: {name}")
            lines.append(f"     Memory: {allocated:.1f}GB used / {total:.1f}GB total ({free:.1f}GB free)")
    else:
        lines.append(f"\n🎮 GPU: ❌ Not available ({gpu_info.get('reason', 'Unknown')})")
    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="DotsOCR Parser Management Client")
    parser.add_argument("--server", default="http://localhost:8000", 
                       help="API server URL (default: http://localhost:8000)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Status command
    subparsers.add_parser("status", help="Get parser status and GPU info")
    
    # Start command
    subparsers.add_parser("start", help="Start/initialize the parser")
    
    # Stop command  
    subparsers.add_parser("stop", help="Stop/destroy the parser and free GPU memory")
    
    # Restart command
    subparsers.add_parser("restart", help="Restart the parser (stop + start)")
    
    # Health command
    subparsers.add_parser("health", help="Check if API server is running")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor parser status continuously")
    monitor_parser.add_argument("--interval", type=int, default=5, 
                               help="Update interval in seconds (default: 5)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    client = ParserManagerClient(args.server)
    
    try:
        if args.command == "status":
            print("📊 Getting parser status...")
            result = client.get_status()
            print(format_status(result))
            
        elif args.command == "start":
            print("🚀 Starting parser...")
            result = client.start_parser()
            if result.get("status") == "success":
                print("✅ Parser started successfully!")
                print(format_status({"status": "success", "data": result.get("data", {})}))
            else:
                print(f"❌ Failed to start parser: {result.get('message', 'Unknown error')}")
                
        elif args.command == "stop":
            print("🛑 Stopping parser...")
            result = client.stop_parser()
            if result.get("status") in ["success", "info"]:
                print(f"✅ {result.get('message', 'Parser stopped')}")
                print(format_status({"status": "success", "data": result.get("data", {})}))
            else:
                print(f"❌ Failed to stop parser: {result.get('message', 'Unknown error')}")
                
        elif args.command == "restart":
            print("🔄 Restarting parser...")
            result = client.restart_parser()
            if result.get("status") == "success":
                print("✅ Parser restarted successfully!")
                print(format_status({"status": "success", "data": result.get("data", {})}))
            else:
                print(f"❌ Failed to restart parser: {result.get('message', 'Unknown error')}")
                
        elif args.command == "health":
            print("🏥 Checking server health...")
            result = client.health_check()
            if result.get("status") == "ok":
                print("✅ API server is running")
                print(f"   Message: {result.get('message', 'N/A')}")
            else:
                print(f"❌ Server health check failed: {result}")
                
        elif args.command == "monitor":
            print(f"👀 Monitoring parser status (updating every {args.interval}s, Ctrl+C to stop)...")
            try:
                while True:
                    # Clear screen (works on most terminals)
                    print("\033[2J\033[H", end="")
                    
                    print(f"📊 DotsOCR Parser Monitor - {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print("=" * 60)
                    
                    result = client.get_status()
                    print(format_status(result))
                    
                    print(f"\n🔄 Next update in {args.interval}s (Ctrl+C to stop)")
                    time.sleep(args.interval)
                    
            except KeyboardInterrupt:
                print("\n👋 Monitoring stopped")
                
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to server at {args.server}")
        print("   Make sure the API server is running")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()