import json
import os
import subprocess
import sys
import time
from pathlib import Path
import importlib.util
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="Chess CV Web App")

# --- Config Management ---
CONFIG_PATH = "config.json"


def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file {CONFIG_PATH} not found.")
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


config = load_config()

ROOT_DIR = Path(__file__).resolve().parent
STOCKFISH_BIND_HOST = config.get("stockfish_host", config.get("host", "0.0.0.0"))
STOCKFISH_CONNECT_HOST = config.get("stockfish_connect_host", STOCKFISH_BIND_HOST)
if STOCKFISH_CONNECT_HOST in {"0.0.0.0", "::", ""}:
    STOCKFISH_CONNECT_HOST = "127.0.0.1"
STOCKFISH_PORT = config.get("stockfish_port", config.get("port", 8000))
WEBAPP_HOST = config.get("webapp_host", config.get("host", "0.0.0.0"))
WEBAPP_PORT = config.get("webapp_port", 8001)
STOCKFISH_URL = f"http://{STOCKFISH_CONNECT_HOST}:{STOCKFISH_PORT}"

VISION_OUTPUT_DIR = Path(config.get("vision_output_dir", "."))
if not VISION_OUTPUT_DIR.is_absolute():
    VISION_OUTPUT_DIR = (ROOT_DIR / VISION_OUTPUT_DIR).resolve()
VISION_CALIBRATION = Path(config.get("vision_calibration", "calibration.npz"))
if not VISION_CALIBRATION.is_absolute():
    VISION_CALIBRATION = (ROOT_DIR / VISION_CALIBRATION).resolve()
VISION_LOG_PATH = Path(config.get("vision_log_path", "vision_server.log"))
if not VISION_LOG_PATH.is_absolute():
    VISION_LOG_PATH = (ROOT_DIR / VISION_LOG_PATH).resolve()
VISION_HEIGHT_THRESHOLD = config.get("vision_height_threshold_mm", 10.0)
VISION_PERCENTAGE_THRESHOLD = config.get("vision_percentage_threshold", 0.4)
VISION_INTERVAL = config.get("vision_interval_seconds", 1.0)
VISION_MAX_FPS = config.get("vision_max_fps", 15.0)
VISION_MAX_PEAKS = config.get("vision_max_peaks", 32)
VISION_OCC_AVG_FRAMES = config.get("vision_occ_avg_frames", 3)
VISION_HEARTBEAT_SECONDS = config.get("vision_heartbeat_seconds", 5.0)
VISION_PROCESS = None


def _forward_stockfish(method, path, payload=None, timeout=6):
    url = f"{STOCKFISH_URL}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return resp.status, resp.headers.get_content_type(), body
    except HTTPError as e:
        body = e.read()
        return e.code, e.headers.get_content_type(), body
    except URLError as e:
        raise HTTPException(status_code=503, detail=f"Stockfish server offline: {e.reason}")


def _stockfish_online():
    try:
        status, _, _ = _forward_stockfish("GET", "/health", timeout=1.5)
        return status == 200
    except HTTPException:
        return False


def _vision_output_path():
    return VISION_OUTPUT_DIR / "occupied_bitmap.npy"


def _vision_online():
    path = _vision_output_path()
    if not path.exists():
        return False
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return age <= VISION_HEARTBEAT_SECONDS


def _tail_file(path: Path, max_lines: int = 30, max_chars: int = 4000) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return ""
    tail = "".join(lines[-max_lines:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail.strip()


# --- Webapp Files ---
app.mount("/static", StaticFiles(directory="webapp"), name="static")


@app.get("/")
def serve_index():
    return FileResponse("webapp/index_v2.html")


@app.get("/health")
def health():
    return {"status": "ok"}


# --- Stockfish control ---
@app.get("/stockfish/health")
def stockfish_health():
    if _stockfish_online():
        return {"status": "online"}
    return JSONResponse(status_code=503, content={"status": "offline"})


@app.post("/stockfish/start")
def start_stockfish():
    if _stockfish_online():
        return {"status": "already_running"}

    server_path = Path(__file__).resolve().parent / "stockfish_server.py"
    if not server_path.exists():
        raise HTTPException(status_code=500, detail="stockfish_server.py not found.")

    subprocess.Popen([sys.executable, str(server_path)], cwd=str(server_path.parent))

    for _ in range(40):
        time.sleep(0.5)
        if _stockfish_online():
            return {"status": "started"}

    raise HTTPException(status_code=500, detail="Stockfish server failed to start in time.")


# --- Vision control ---


@app.post("/vision/service/capture")
def capture_vision_service(frames: int = 15):
    script_path = Path(__file__).resolve().parent / "chess vision" / "detect_peaks_service.py"
    if not script_path.exists():
        raise HTTPException(status_code=500, detail="detect_peaks_service.py not found.")
        
    output_path = Path(__file__).resolve().parent / "service_output.json"
    
    args = [
        sys.executable,
        str(script_path),
        "--num-frames", str(frames),
        "--no-visualize",
        "--calibration", str(VISION_CALIBRATION),
        "--json-out", str(output_path)
    ]
    
    try:
        res = subprocess.run(args, check=True, cwd=str(script_path.parent), capture_output=True, text=True)
        with open(output_path, "r") as f:
            data = json.load(f)
        return data
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Vision service failed. Error: {e.stderr}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Stockfish proxy endpoints ---
@app.get("/stockfish/camera_occupancy")
def proxy_camera_occupancy():
    status, content_type, body = _forward_stockfish("GET", "/camera_occupancy")
    return Response(body, status_code=status, media_type=content_type)


@app.post("/stockfish/infer_move")
def proxy_infer_move(payload: dict):
    status, content_type, body = _forward_stockfish("POST", "/infer_move", payload)
    return Response(body, status_code=status, media_type=content_type)


@app.post("/stockfish/best_move")
def proxy_best_move(payload: dict):
    status, content_type, body = _forward_stockfish("POST", "/best_move", payload)
    return Response(body, status_code=status, media_type=content_type)


@app.post("/stockfish/rate_move")
def proxy_rate_move(payload: dict):
    status, content_type, body = _forward_stockfish("POST", "/rate_move", payload)
    return Response(body, status_code=status, media_type=content_type)


if __name__ == "__main__":
    print(f"Starting web app on {WEBAPP_HOST}:{WEBAPP_PORT}")
    uvicorn.run("webapp_server_v2:app", host=WEBAPP_HOST, port=WEBAPP_PORT, reload=True)
