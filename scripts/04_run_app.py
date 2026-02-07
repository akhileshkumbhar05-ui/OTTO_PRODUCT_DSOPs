import subprocess
import sys

def main():
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "app/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
    ]
    print("[INFO] Running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
