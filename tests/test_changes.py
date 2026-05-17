import numpy as np
from chess_move_inference.changes import find_changes

def test_find_changes():
    old = np.zeros((8, 8))
    new = np.zeros((8, 8))
    old[0, 0] = 1.0
    new[0, 0] = 0.0 # became empty
    
    old[1, 1] = 0.0
    new[1, 1] = 1.0 # became filled
    
    old[2, 2] = 1.0
    new[2, 2] = 1.0 # stayed filled
    
    became_empty, became_filled, stayed_filled = find_changes(old, new)
    assert became_empty == {(0, 0)}
    assert became_filled == {(1, 1)}
    assert stayed_filled == {(2, 2)}
