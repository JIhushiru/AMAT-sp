"""Launch both FastAPI backend and React frontend with a single command."""

import subprocess
import sys
import os
import signal
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
API_DIR = os.path.join(ROOT, "webapp", "api")
FRONTEND_DIR = os.path.join(ROOT, "webapp", "frontend")

def main():
    procs = []

    try:
        # Start FastAPI backend
        print("[*] Starting FastAPI backend on http://localhost:8000 ...")
        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--reload", "--port", "8000"],
            cwd=API_DIR,
        )
        procs.append(backend)

        # Install frontend deps if needed
        node_modules = os.path.join(FRONTEND_DIR, "node_modules")
        if not os.path.isdir(node_modules):
            print("[*] Installing frontend dependencies ...")
            subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, shell=True, check=True)

        # Start React dev server
        print("[*] Starting React frontend on http://localhost:5173 ...")
        frontend = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=FRONTEND_DIR,
            shell=True,
        )
        procs.append(frontend)

        print("\n[OK] Both servers are running. Press Ctrl+C to stop.\n")

        # Wait for either process to exit
        while all(p.poll() is None for p in procs):
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[*] Shutting down ...")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            p.wait()
        print("[*] Stopped.")


if __name__ == "__main__":
    main()
