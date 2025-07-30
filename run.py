#!/usr/bin/env python3
"""
Launcher script for the MCP Streamlit Interface
"""
import subprocess
import sys
import os

def main():
    """Launch the Streamlit interface"""
    try:
        # Change to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Run streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_ui.py", "--server.port", "8503"]
        
        print("🚀 Starting MCP Streamlit Interface...")
        print("📱 Interface will be available at: http://localhost:8503")
        print("🛑 Press Ctrl+C to stop the server")
        print("-" * 50)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down the interface...")
    except Exception as e:
        print(f"❌ Error starting interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
