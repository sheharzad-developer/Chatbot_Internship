#!/usr/bin/env python3
"""
Launch the new HTML frontend
"""
import subprocess
import sys
import requests
import time
from pathlib import Path

def check_backend():
    """Check if the backend is running"""
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running and healthy")
            return True
        else:
            print(f"⚠️  Backend responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("❌ Backend is not running or not accessible")
        return False

def main():
    print("🚀 Starting Reflect Agent AI Frontend")
    print("=" * 50)
    
    # Check if backend is running
    print("🔍 Checking backend status...")
    if not check_backend():
        print("\n⚠️  Backend not detected. Please start the backend first:")
        print("   uvicorn main:app --host 0.0.0.0 --port 8001")
        print("\nContinuing to start frontend anyway...")
        time.sleep(2)
    
    # Start the HTML frontend server
    frontend_script = Path(__file__).parent / "frontend" / "server.py"
    
    try:
        print(f"\n🌐 Starting HTML frontend server...")
        subprocess.run([sys.executable, str(frontend_script)], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Frontend stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting frontend: {e}")
    except FileNotFoundError:
        print(f"❌ Frontend server script not found: {frontend_script}")

if __name__ == "__main__":
    main() 