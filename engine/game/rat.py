import random
import bisect
from typing import Tuple
from .enums import Cell, Noise, BOARD_SIZE

HEADSTART_MOVES = 1000

# Noise probabilities based on cell type
# [P(squeak), P(scratch), P(squeal)]
NOISE_PROBS = {
    Cell.BLOCKED: (0.5, 0.3, 0.2),
    Cell.SPACE: (0.7, 0.15, 0.15),
    Cell.PRIMED: (0.1, 0.8, 0.1),
    Cell.CARPET: (0.1, 0.1, 0.8),
}

# [P(-1), P(correct), P(+1), P(+2)]
DISTANCE_ERROR_PROBS = (0.12, 0.7, 0.12, 0.06)
DISTANCE_ERROR_OFFSETS = (-1, 0, 1, 2)


def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def cumulative(probs):
    c = []
    s = 0.0
    for p in probs:
        s += p
        c.append(s)
    return c


class Rat:
    def __init__(self, T):
        """
        T[from_index, to_index] = transition probability
        """

        num_positions = BOARD_SIZE * BOARD_SIZE

        # build cumulative transition matrix (CPU floats)
        self.cumT = []
        for i in range(num_positions):
            running_sum = 0.0
            row = []
            for p in T[i]:
                running_sum += float(p)
                row.append(running_sum)
            self.cumT.append(row)

        self.noise_cum = {k: cumulative(v) for k, v in NOISE_PROBS.items()}
        self.dist_cum = cumulative(DISTANCE_ERROR_PROBS)

        self.position = (0, 0)

    def _pos_to_index(self, pos: Tuple[int, int]) -> int:
        return pos[1] * BOARD_SIZE + pos[0]

    def _index_to_pos(self, index: int) -> Tuple[int, int]:
        y = index // BOARD_SIZE
        x = index % BOARD_SIZE
        return (x, y)

    def _sample3(self, cum):
        r = random.random()
        if r < cum[0]:
            return 0
        if r < cum[1]:
            return 1
        return 2

    def move(self):
        from_index = self._pos_to_index(self.position)
        r = random.random()
        cum_probs = self.cumT[from_index]

        # binary search instead of slow loop
        to_index = bisect.bisect_left(cum_probs, r)

        self.position = self._index_to_pos(to_index)

    def make_noise(self, board) -> Noise:
        cell_type = board.get_cell(self.position)
        cum = self.noise_cum.get(cell_type, self.noise_cum[Cell.SPACE])
        idx = self._sample3(cum)
        return Noise(idx)

    def estimate_distance(self, worker_position: Tuple[int, int]) -> int:
        actual = manhattan_distance(worker_position, self.position)

        r = random.random()
        offset = DISTANCE_ERROR_OFFSETS[-1]
        for i, threshold in enumerate(self.dist_cum):
            if r < threshold:
                offset = DISTANCE_ERROR_OFFSETS[i]
                break

        d = actual + offset
        return d if d > 0 else 0

    def spawn(self):
        self.position = (0, 0)
        for _ in range(HEADSTART_MOVES):
            self.move()

    def get_position(self) -> Tuple[int, int]:
        return self.position

    def sample(self, board):
        return (
            self.make_noise(board),
            self.estimate_distance(board.player_worker.get_location()),
        )
