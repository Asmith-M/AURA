import uvicorn
import argparse
import sys
import os
from sentinel.config.sentinel_config import sentinel_config

def main():
    """Start the Aura Sentinel API server"""
    
    parser = argparse.ArgumentParser(description="Start Aura Sentinel API")
    parser.add_argument("--host", type=str, default=sentinel_config.API_HOST,
                       help="Host address for the API server")
    parser.add_argument("--port", type=int, default=sentinel_config.API_PORT,
                       help="Port for the API server")
    parser.add_argument("--reload", action="store_true",
                       help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print(f"Starting Aura Sentinel API server...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Log Level: {sentinel_config.API_LOG_LEVEL}")
    print(f"Models Directory: {sentinel_config.MODELS_DIR}")
    print(f"Reports Directory: {sentinel_config.REPORTS_DIR}")
    
    # Create necessary directories
    os.makedirs(sentinel_config.MODELS_DIR, exist_ok=True)
    os.makedirs(sentinel_config.REPORTS_DIR, exist_ok=True)
    os.makedirs(sentinel_config.FINGERPRINTS_DIR, exist_ok=True)
    
    print("Directories created/verified")
    
    # Start the server
    uvicorn.run(
        "sentinel.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=sentinel_config.API_LOG_LEVEL
    )

if __name__ == "__main__":
    main()