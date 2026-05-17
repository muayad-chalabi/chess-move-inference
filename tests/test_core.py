import numpy as np
import chess
from chess_move_inference.core import ChessMoveInference
from chess_move_inference.utils import board_to_occupancy

def test_inference_e2e4():
    inference = ChessMoveInference()
    board = chess.Board()
    
    # create e2e4 board
    board_next = board.copy()
    board_next.push_san("e4")
    new_occ = board_to_occupancy(board_next)
    
    best_move, score = inference.infer_move(board, new_occ)
    assert best_move == "e2e4"
    assert score > 0.5
