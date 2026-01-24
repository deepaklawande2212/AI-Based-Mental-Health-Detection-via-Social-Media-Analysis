# Development Server Startup Script
# This script helps you start the Depression Detection System backend server

import os
import sys
import subprocess
from pathlib import Path

def main():
    """
    Start the FastAPI development server
    """
    print("🧠 Depression Detection System Backend")
    print("=" * 50)
    
    # Check if we're in the correct directory
    current_dir = Path.cwd()
    if not (current_dir / "app" / "main.py").exists():
        print("❌ Error: Please run this script from the server directory")
        print(f"Current directory: {current_dir}")
        print("Expected: ../server/")
        return 1
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: No virtual environment detected")
        print("Recommendation: Create and activate a virtual environment first")
        print()
    
    # Check if requirements are installed
    try:
        import fastapi
        import uvicorn
        print("✅ FastAPI and dependencies found")
    except ImportError:
        print("❌ Error: Required packages not installed")
        print("Please run: pip install -r requirements.txt")
        return 1
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    print(f"📁 Logs directory: {logs_dir.absolute()}")
    
    print()
    print("🚀 Starting development server...")
    print("📍 Server will be available at:")
    print("   - Local: http://localhost:8000")
    print("   - Network: http://0.0.0.0:8000")
    print("📚 API Documentation:")
    print("   - Swagger UI: http://localhost:8000/docs")
    print("   - ReDoc: http://localhost:8000/redoc")
    print()
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the server
    try:
        os.environ["PYTHONPATH"] = str(current_dir)
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server failed to start: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
