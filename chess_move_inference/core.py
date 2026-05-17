import numpy as np
import chess
from typing import Tuple, Optional

from .utils import board_to_occupancy
from .changes import find_changes
from .hypotheses import infer_moves, infer_captures, infer_castles, infer_en_passant, infer_nothing
from .scoring import score_hypothesis

class ChessMoveInference:
    def __init__(self):
        self.priors = {
            'move': 0.90,
            'capture': 0.85,
            'castle': 0.55,
            'en_passant': 0.20,
            'nothing': 0.01,
        }
        self.illegal_penalty = 0.1
        
    def infer_move(self, board_state: chess.Board, occupancy_map: np.ndarray) -> Tuple[Optional[str], float]:
        old_occupancy = board_to_occupancy(board_state)
        became_empty, became_filled, stayed_filled = find_changes(old_occupancy, occupancy_map)
        
        move_hyps = infer_moves(board_state, became_empty, became_filled, stayed_filled)
        capture_hyps = infer_captures(board_state, became_empty, became_filled, stayed_filled)
        castle_hyps = infer_castles(board_state, became_empty, became_filled, stayed_filled)
        en_passant_hyps = infer_en_passant(board_state, became_empty, became_filled, stayed_filled)
        nothing_hyps = infer_nothing(board_state, became_empty, became_filled, stayed_filled)
        
        all_hypotheses = []
        
        for hyp in move_hyps:
            score = score_hypothesis(hyp, 'move', board_state, old_occupancy, occupancy_map, self.priors, self.illegal_penalty)
            all_hypotheses.append((hyp, 'move', score))
            
        for hyp in capture_hyps:
            score = score_hypothesis(hyp, 'capture', board_state, old_occupancy, occupancy_map, self.priors, self.illegal_penalty)
            all_hypotheses.append((hyp, 'capture', score))
            
        for hyp in castle_hyps:
            score = score_hypothesis(hyp, 'castle', board_state, old_occupancy, occupancy_map, self.priors, self.illegal_penalty)
            all_hypotheses.append((hyp, 'castle', score))
            
        for hyp in en_passant_hyps:
            score = score_hypothesis(hyp, 'en_passant', board_state, old_occupancy, occupancy_map, self.priors, self.illegal_penalty)
            all_hypotheses.append((hyp, 'en_passant', score))
            
        for hyp in nothing_hyps:
            score = score_hypothesis(hyp, 'nothing', board_state, old_occupancy, occupancy_map, self.priors, self.illegal_penalty)
            all_hypotheses.append((hyp, 'nothing', score))
            
        if not all_hypotheses:
            return None, 0.0
            
        best_hyp, _, best_score = max(all_hypotheses, key=lambda x: x[2])
        return best_hyp, best_score
