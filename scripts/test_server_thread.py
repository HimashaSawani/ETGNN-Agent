import sys
import os
import time
import threading
import urllib.request

sys.path.insert(0, os.path.abspath("."))

import uvicorn
from src.web.app import app

class ThreadedUvicornServer(uvicorn.Server):
    def install_signal_handlers(self):
        # Override signal handlers so Uvicorn can run inside a worker thread
        pass

def main():
    print("Initializing Threaded Uvicorn Server...", flush=True)
    config = uvicorn.Config(app=app, host="127.0.0.1", port=8000, log_level="warning")
    server = ThreadedUvicornServer(config=config)
    
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    
    print("Thread started, waiting for server.started ...", flush=True)
    for i in range(15):
        time.sleep(0.5)
        if server.started:
            print(f"Server is started! ({i*0.5:.1f}s)", flush=True)
            break
            
    # Probe health
    req = urllib.request.Request("http://127.0.0.1:8000/api/health")
    with urllib.request.urlopen(req, timeout=5) as resp:
        print("Health Response:", resp.status, resp.read().decode("utf-8"), flush=True)
        
    server.should_exit = True
    print("Server cleanly stopped!", flush=True)

if __name__ == "__main__":
    main()
