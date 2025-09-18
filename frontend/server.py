#!/usr/bin/env python3
"""
Simple HTTP server for testing the Tennis Tracker frontend.
This is just for development/testing purposes.
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)
    
    def end_headers(self):
        # Add CORS headers for development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    PORT = 8080
    
    print(f"🎾 Tennis Tracker Frontend Server")
    print(f"📁 Serving from: {Path(__file__).parent}")
    print(f"🌐 Server running at: http://localhost:{PORT}")
    print(f"📖 Open http://localhost:{PORT}/index.html in your browser")
    print(f"⚠️  Note: This is a development server. API endpoints are not implemented yet.")
    print(f"🛑 Press Ctrl+C to stop the server")
    
    try:
        with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
            # Automatically open browser
            webbrowser.open(f'http://localhost:{PORT}/index.html')
            httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n👋 Server stopped")

if __name__ == "__main__":
    main()
