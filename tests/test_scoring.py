import numpy as np
import chess
from chess_move_inference.scoring import score_hypothesis

def test_score_hypothesis():
    board = chess.Board()
    old_occ = np.zeros((8, 8))
    new_occ = np.zeros((8, 8))
    
    priors = {'move': 0.6}
    penalty = 0.01
    
    # Needs actual occupancy for e2e4 to get a high score
    # Let's mock a perfect occupancy change
    from_idx = (6, 4) # e2
    to_idx = (4, 4)   # e4
    new_occ[from_idx] = 0.0 # empty
    new_occ[to_idx] = 1.0   # filled
    
    score = score_hypothesis("e2e4", "move", board, old_occ, new_occ, priors, penalty)
    assert score > 0.5
