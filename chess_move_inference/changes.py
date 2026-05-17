import numpy as np
from typing import Tuple, Set

def find_changes(old_occupancy: np.ndarray, new_occupancy: np.ndarray, threshold: float = 0.5) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    old_occupied = old_occupancy > threshold
    new_occupied = new_occupancy > threshold
    
    became_empty = set(map(tuple, np.argwhere(old_occupied & ~new_occupied)))
    became_filled = set(map(tuple, np.argwhere(~old_occupied & new_occupied)))
    stayed_filled = set(map(tuple, np.argwhere(old_occupied & new_occupied)))
    
    return became_empty, became_filled, stayed_filled
