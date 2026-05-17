import chess
import requests
import sys
import time

SERVER_URL = "http://localhost:8000"

def play_game(max_full_moves=50):
    board = chess.Board()
    
    print("=== Starting Automated Game Against Stockfish Server ===\n")
    print(board)
    print("\n" + "="*40 + "\n")
    
    for full_move in range(1, max_full_moves + 1):
        for color in [chess.WHITE, chess.BLACK]:
            if board.is_game_over():
                print(f"Game over! Result: {board.result()}")
                return

            fen = board.fen()
            
            # --- 1. Test /best_move endpoint ---
            try:
                best_resp = requests.post(f"{SERVER_URL}/best_move", json={"fen": fen})
                best_resp.raise_for_status()
                best_move = best_resp.json().get("best_move")
            except requests.exceptions.RequestException as e:
                print(f"ERROR: Failed to connect to server at {SERVER_URL}/best_move")
                print(f"Details: {e}")
                sys.exit(1)
                
            if not best_move:
                print("No valid moves returned. Game over.")
                return

            # --- 2. Test /rate_move endpoint ---
            try:
                rate_resp = requests.post(f"{SERVER_URL}/rate_move", json={"fen": fen, "move": best_move})
                rate_resp.raise_for_status()
                rate_data = rate_resp.json()
            except requests.exceptions.RequestException as e:
                print(f"ERROR: Failed to connect to server at {SERVER_URL}/rate_move")
                print(f"Details: {e}")
                sys.exit(1)
            
            # --- 3. Process and Print ---
            move_obj = chess.Move.from_uci(best_move)
            san_move = board.san(move_obj)
            board.push(move_obj)
            
            player = "White" if color == chess.WHITE else "Black"
            
            # Convert centipawns to standard pawn evaluation metric (e.g. +1.50)
            eval_cp = rate_data.get('eval_after_centipawns', 0)
            eval_score = eval_cp / 100.0
            
            rating = rate_data.get('rating', 'Unknown')
            
            print(f"Move {full_move} ({player}): {san_move}")
            print(f"Engine Rating: {rating} | Eval: {'+' if eval_score > 0 else ''}{eval_score:.2f}")
            print(board)
            print("-" * 40)
            
            # Optional: Sleep briefly to avoid hammering the server too quickly
            time.sleep(0.1)

    print(f"\nReached move limit of {max_full_moves} full moves. Stopping game.")

if __name__ == "__main__":
    # Ensure the FastAPI server is running before executing this script
    play_game(max_full_moves=50)
