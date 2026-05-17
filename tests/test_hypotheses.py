import chess
from chess_move_inference.hypotheses import infer_moves, infer_captures, infer_castles
from chess_move_inference.utils import square_to_idx

def test_infer_moves():
    board = chess.Board()
    from_idx = square_to_idx(chess.E2)
    to_idx = square_to_idx(chess.E4)
    hyps = infer_moves(board, {from_idx}, {to_idx}, set())
    assert "e2e4" in hyps

def test_infer_captures():
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    from_idx = square_to_idx(chess.D2)
    target_idx = square_to_idx(chess.E5)
    hyps = infer_captures(board, {from_idx}, set(), {target_idx})
    assert "d2e5" in hyps

def test_infer_castles():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3")
    king_idx = square_to_idx(chess.E1)
    rook_idx = square_to_idx(chess.H1)
    king_to_idx = square_to_idx(chess.G1)
    rook_to_idx = square_to_idx(chess.F1)
    
    hyps = infer_castles(board, {king_idx, rook_idx}, {king_to_idx, rook_to_idx}, set())
    assert "e1g1" in hyps or "h1f1" in hyps
