import os
import time
from threading import Thread

# Set port for FastAPI (internal)
FASTAPI_PORT = 8000
# Hugging Face Spaces uses port 7860 by default
STREAMLIT_PORT = 7860

def run_fastapi():
    """Run the FastAPI server on internal port"""
    os.system(f"uvicorn main:app --host 0.0.0.0 --port {FASTAPI_PORT}")

def run_streamlit():
    """Run the Streamlit app on the Hugging Face port"""
    os.system(f"streamlit run streamlit_app.py --server.port {STREAMLIT_PORT} --server.address 0.0.0.0")

if __name__ == "__main__":
    # Start FastAPI server in a separate thread
    print("Starting FastAPI server...")
    api_thread = Thread(target=run_fastapi)
    api_thread.daemon = True
    api_thread.start()
    
    # Give FastAPI time to start
    time.sleep(5)
    
    # Start Streamlit in the main thread
    print("Starting Streamlit frontend...")
    run_streamlit()

