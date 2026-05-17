import json
import os
import chess
import chess.engine
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from typing import List
from chess_move_inference import ChessMoveInference

app = FastAPI(title="Stockfish Chess Server")

# Allow CORS so external UI/software can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config Management ---
CONFIG_PATH = "config.json"

def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file {CONFIG_PATH} not found.")
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

config = load_config()

STOCKFISH_PATH = config.get("stockfish_path", "./stockfish/stockfish.exe")
THINK_TIME = config.get("time_to_think_seconds", 0.5)
SKILL_LEVEL = config.get("skill_level", 20)

# Validate Stockfish path
if not os.path.exists(STOCKFISH_PATH):
    print(f"\nWARNING: Stockfish executable not found at '{STOCKFISH_PATH}'.")
    print("Please download Stockfish (https://stockfishchess.org/download/),")
    print("extract it, and update the 'stockfish_path' in config.json.\n")

# Provide inference model globally
inference_model = ChessMoveInference()

# --- Pydantic Models ---
class BestMoveRequest(BaseModel):
    fen: str

class RateMoveRequest(BaseModel):
    fen: str
    move: str

class InferMoveRequest(BaseModel):
    fen: str
    occupancy_map: List[List[float]]

# --- Helper Functions ---
def get_engine():
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine.configure({"Skill Level": SKILL_LEVEL})
        return engine
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start Stockfish engine: {e}")


# Path where chess vision run.py writes its output (configurable via config.json)
VISION_OUTPUT_DIR = config.get("vision_output_dir", ".")
OCCUPIED_BITMAP_PATH = os.path.join(VISION_OUTPUT_DIR, "occupied_bitmap.npy")


# --- API Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/camera_occupancy")
def camera_occupancy():
    """
    Read the latest occupied_bitmap.npy written by chess vision run.py.
    Returns an 8x8 grid of 0.0/1.0 values as a JSON list of lists.
    """
    if not os.path.exists(OCCUPIED_BITMAP_PATH):
        raise HTTPException(
            status_code=404,
            detail=f"occupied_bitmap.npy not found at '{OCCUPIED_BITMAP_PATH}'. "
                   "Make sure chess vision run.py is running."
        )
    try:
        bitmap = np.load(OCCUPIED_BITMAP_PATH).astype(np.float32)
        if bitmap.shape != (8, 8):
            raise HTTPException(status_code=500, detail=f"Unexpected bitmap shape: {bitmap.shape}")
        return {"occupancy_map": bitmap.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read occupancy bitmap: {e}")

@app.post("/infer_move")
def infer_move(req: InferMoveRequest):
    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN string")

    # Map the webapp's nested 8x8 lists into a numpy array for the Python library
    occ_map = np.array(req.occupancy_map)
    best_move, confidence = inference_model.infer_move(board, occ_map)

    return {
        "best_move": best_move,
        "confidence": float(confidence)
    }

@app.post("/best_move")
def get_best_move(req: BestMoveRequest):
    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN string")

    engine = get_engine()
    try:
        result = engine.play(board, chess.engine.Limit(time=THINK_TIME))
        return {
            "fen": req.fen,
            "best_move": result.move.uci() if result.move else None
        }
    finally:
        engine.quit()

@app.post("/rate_move")
def rate_move(req: RateMoveRequest):
    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN string")

    try:
        move = chess.Move.from_uci(req.move)
        if move not in board.legal_moves:
            raise HTTPException(status_code=400, detail="Illegal move for the given board state")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UCI move format")

    engine = get_engine()
    try:
        info_before = engine.analyse(board, chess.engine.Limit(time=THINK_TIME))
        score_before = info_before["score"].pov(board.turn)
        
        board.push(move)
        info_after = engine.analyse(board, chess.engine.Limit(time=THINK_TIME))
        score_after = info_after["score"].pov(not board.turn)

        cp_before = score_before.score(mate_score=10000)
        cp_after = score_after.score(mate_score=10000)

        diff = cp_after - cp_before

        rating = "Good"
        if diff < -300:
            rating = "Blunder"
        elif diff < -100:
            rating = "Mistake"
        elif diff < -50:
            rating = "Inaccuracy"
        elif diff > 50:
            rating = "Brilliant"

        return {
            "eval_before_centipawns": cp_before,
            "eval_after_centipawns": cp_after,
            "difference": diff,
            "rating": rating
        }
    finally:
        engine.quit()

# --- Serve Webapp Files ---
app.mount("/static", StaticFiles(directory="webapp"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("webapp/index.html")

if __name__ == "__main__":
    host = config.get("host", "0.0.0.0")
    port = config.get("port", 8000)
    print(f"Starting server on {host}:{port}")
    uvicorn.run("stockfish_server:app", host=host, port=port, reload=True)
