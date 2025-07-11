#!/usr/bin/env python3
"""
Run the Gradio frontend for the Reflect Agent Chatbot
"""
import subprocess
import sys
import time
import requests

def check_backend():
    """Check if the FastAPI backend is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("🤖 Reflect Agent Chatbot - Frontend Launcher")
    print("=" * 50)
    
    # Check if backend is running
    if not check_backend():
        print("⚠️  FastAPI backend is not running!")
        print("Please start the backend first:")
        print("   python3 main.py")
        print("   OR")
        print("   python3 run.py")
        print()
        
        choice = input("Start backend automatically? (y/n): ").lower().strip()
        if choice == 'y':
            print("🚀 Starting FastAPI backend...")
            subprocess.Popen([sys.executable, "main.py"])
            
            # Wait for backend to start
            print("⏳ Waiting for backend to start...")
            for i in range(30):  # Wait up to 30 seconds
                if check_backend():
                    print("✅ Backend is ready!")
                    break
                time.sleep(1)
                print(f"   Checking... ({i+1}/30)")
            else:
                print("❌ Backend failed to start. Please start manually.")
                return
        else:
            print("❌ Frontend requires backend to be running. Exiting.")
            return
    else:
        print("✅ Backend is running!")
    
    print()
    print("🌐 Starting Gradio frontend...")
    print("📱 Frontend will open at: http://localhost:7860")
    print("📚 Backend API docs at: http://localhost:8000/docs")
    print()
    
    # Start the frontend
    from frontend.gradio_app import create_interface
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )

if __name__ == "__main__":
    main() 