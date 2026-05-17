import numpy as np
import chess
from typing import Tuple

def board_to_occupancy(board: chess.Board) -> np.ndarray:
    occupancy = np.zeros((8, 8))
    for square in chess.SQUARES:
        if board.piece_at(square) is not None:
            row, col = divmod(square, 8)
            occupancy[7 - row, col] = 1.0
    return occupancy

def idx_to_square(idx: Tuple[int, int]) -> int:
    row, col = idx
    return (7 - row) * 8 + col

def square_to_idx(square: int) -> Tuple[int, int]:
    row = square // 8
    col = square % 8
    return (7 - row, col)
