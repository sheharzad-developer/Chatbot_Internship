#!/usr/bin/env python3
"""
Simple HTTP server to serve the HTML frontend
"""
import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

# Change to frontend directory
frontend_dir = Path(__file__).parent
os.chdir(frontend_dir)

PORT = 3000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow communication with backend
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

if __name__ == "__main__":
    try:
        with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
            print(f"🌐 Frontend server starting at http://localhost:{PORT}")
            print(f"📁 Serving files from: {frontend_dir}")
            print("🔗 Make sure your backend is running on http://localhost:8001")
            print("\n🚀 Opening browser...")
            
            # Try to open browser
            try:
                webbrowser.open(f'http://localhost:{PORT}')
            except:
                pass
            
            print(f"\n✅ Frontend ready! Visit: http://localhost:{PORT}")
            print("Press Ctrl+C to stop")
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {PORT} is already in use. Try a different port or stop the other service.")
        else:
            print(f"❌ Error starting server: {e}") 