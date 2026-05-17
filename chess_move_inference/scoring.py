import numpy as np
import chess
from typing import Dict
from .utils import square_to_idx

def score_hypothesis(move_uci: str, scenario: str, board: chess.Board, 
                     old_occupancy: np.ndarray, new_occupancy: np.ndarray,
                     priors: Dict[str, float], illegal_penalty: float) -> float:
    score = priors.get(scenario, 0.0)
    
    if move_uci == "0000":
        # Nothing moved scenario
        diff = np.abs(old_occupancy - new_occupancy)
        mean_error = float(np.mean(diff))
        occupancy_confidence = max(0.0, 1.0 - mean_error * 2) 
        score *= (0.7 + 0.3 * occupancy_confidence)
        return score
        
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            score *= illegal_penalty
    except ValueError:
        return 0.0
        
    move = chess.Move.from_uci(move_uci)
    from_square = move.from_square
    to_square = move.to_square
    
    from_idx = square_to_idx(from_square)
    to_idx = square_to_idx(to_square)
    
    from_confidence = 1.0 - new_occupancy[from_idx[0], from_idx[1]]
    to_confidence = new_occupancy[to_idx[0], to_idx[1]]
    
    occupancy_confidence = (from_confidence + to_confidence) / 2
    score *= (0.7 + 0.3 * occupancy_confidence)
    
    return score
