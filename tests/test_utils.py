import numpy as np
import chess
from chess_move_inference.utils import board_to_occupancy, idx_to_square, square_to_idx

def test_board_to_occupancy():
    board = chess.Board()
    occ = board_to_occupancy(board)
    assert occ.shape == (8, 8)
    assert occ[0, 0] == 1.0  # a1
    assert occ[3, 3] == 0.0  # d4 (empty)

def test_idx_to_square():
    assert idx_to_square((7, 0)) == chess.A1
    assert idx_to_square((0, 7)) == chess.H8

def test_square_to_idx():
    assert square_to_idx(chess.A1) == (7, 0)
    assert square_to_idx(chess.H8) == (0, 7)
