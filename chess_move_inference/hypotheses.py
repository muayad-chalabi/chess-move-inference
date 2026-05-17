import chess
import itertools
from typing import List, Set, Tuple
from .utils import idx_to_square

def infer_moves(board: chess.Board, became_empty: Set[Tuple[int, int]], became_filled: Set[Tuple[int, int]], stayed_filled: Set[Tuple[int, int]]) -> List[str]:
    hypotheses = []
    if len(became_empty) == 1 and len(became_filled) == 1:
        from_square_idx = list(became_empty)[0]
        to_square_idx = list(became_filled)[0]
        
        from_square = idx_to_square(from_square_idx)
        to_square = idx_to_square(to_square_idx)
        
        move = chess.Move(from_square, to_square)
        if board.piece_at(from_square) and board.piece_at(from_square).piece_type == chess.PAWN:
            if (board.piece_at(from_square).color == chess.WHITE and to_square >= 56) or \
               (board.piece_at(from_square).color == chess.BLACK and to_square <= 7):
                move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
        
        hypotheses.append(move.uci())
    return hypotheses

def infer_captures(board: chess.Board, became_empty: Set[Tuple[int, int]], became_filled: Set[Tuple[int, int]], stayed_filled: Set[Tuple[int, int]]) -> List[str]:
    hypotheses = []
    if len(became_empty) == 1 and len(became_filled) == 0 and len(stayed_filled) >= 1:
        from_square_idx = list(became_empty)[0]
        from_square = idx_to_square(from_square_idx)
        
        for to_square_idx in stayed_filled:
            to_square = idx_to_square(to_square_idx)
            move = chess.Move(from_square, to_square)
            
            if board.piece_at(from_square) and board.piece_at(from_square).piece_type == chess.PAWN:
                if (board.piece_at(from_square).color == chess.WHITE and to_square >= 56) or \
                   (board.piece_at(from_square).color == chess.BLACK and to_square <= 7):
                    move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
            
            hypotheses.append(move.uci())
    return hypotheses

def infer_castles(board: chess.Board, became_empty: Set[Tuple[int, int]], became_filled: Set[Tuple[int, int]], stayed_filled: Set[Tuple[int, int]]) -> List[str]:
    hypotheses = []
    if len(became_empty) == 2 and len(became_filled) == 2:
        empty_squares = [idx_to_square(idx) for idx in became_empty]
        filled_squares = [idx_to_square(idx) for idx in became_filled]
        
        for k_from, r_from in itertools.permutations(empty_squares):
            for k_to, r_to in itertools.permutations(filled_squares):
                king_dist = abs(chess.square_file(k_from) - chess.square_file(k_to))
                rook_dist = abs(chess.square_file(r_from) - chess.square_file(r_to))
                
                # Ensure the King's origin is always from the 'E' file (file index 4) 
                # so it generates O-O instead of the Rook's move
                if king_dist == 2 and rook_dist >= 2 and chess.square_file(k_from) == 4:
                    move = chess.Move(k_from, k_to)
                    hypotheses.append(move.uci())
    return hypotheses

def infer_en_passant(board: chess.Board, became_empty: Set[Tuple[int, int]], became_filled: Set[Tuple[int, int]], stayed_filled: Set[Tuple[int, int]]) -> List[str]:
    hypotheses = []
    if len(became_empty) == 2 and len(became_filled) == 1:
        empty_squares = [idx_to_square(idx) for idx in became_empty]
        filled_square = idx_to_square(list(became_filled)[0])
        
        # We append both from_square candidates; illegal ones will be penalized in scoring
        for from_square in empty_squares:
            move = chess.Move(from_square, filled_square)
            hypotheses.append(move.uci())
            
    return hypotheses

def infer_nothing(board: chess.Board, became_empty: Set[Tuple[int, int]], became_filled: Set[Tuple[int, int]], stayed_filled: Set[Tuple[int, int]]) -> List[str]:
    # Always propose "0000" (null move) as a hypothesis for nothing moving
    return ["0000"]
