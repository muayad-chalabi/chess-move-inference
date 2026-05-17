import json
import os
import chess
import chess.engine
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Stockfish Chess Server")

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

# --- Pydantic Models ---
class BestMoveRequest(BaseModel):
    fen: str  # FEN strings inherently encode the board state and whose turn it is

class RateMoveRequest(BaseModel):
    fen: str
    move: str  # UCI string (e.g., "e2e4")

# --- Helper Functions ---
def get_engine():
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine.configure({"Skill Level": SKILL_LEVEL})
        return engine
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start Stockfish engine: {e}")

# --- API Endpoints ---
@app.post("/best_move")
def get_best_move(req: BestMoveRequest):
    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN string")

    engine = get_engine()
    try:
        # Ask Stockfish for the best move
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
        # 1. Evaluate the position BEFORE the move
        info_before = engine.analyse(board, chess.engine.Limit(time=THINK_TIME))
        
        # We look at the score from the perspective of the player whose turn it was
        score_before = info_before["score"].pov(board.turn)
        
        # 2. Evaluate the position AFTER the move
        board.push(move)
        info_after = engine.analyse(board, chess.engine.Limit(time=THINK_TIME))
        
        # After the move, the turn has changed! But we still want to judge 
        # how good the move was for the player who originally made it.
        # So we use `not board.turn` (which refers to the player who just moved).
        score_after = info_after["score"].pov(not board.turn)

        # 3. Calculate Centipawn values
        cp_before = score_before.score(mate_score=10000)
        cp_after = score_after.score(mate_score=10000)

        diff = cp_after - cp_before

        # Determine rating description
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


if __name__ == "__main__":
    host = config.get("host", "0.0.0.0")
    port = config.get("port", 8000)
    print(f"Starting server on {host}:{port}")
    uvicorn.run("stockfish_server:app", host=host, port=port, reload=True)
