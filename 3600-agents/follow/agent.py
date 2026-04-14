"""
follow agent — Never primes. Moves toward the opponent's primed lines and
carpets them for points. Searches for the rat when confident.
"""

from collections.abc import Callable
from typing import Tuple

from game.board import Board
from game.move import Move
from game.enums import (
    MoveType,
    Direction,
    BOARD_SIZE,
    loc_after_direction,
)

_DIRS = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
_DD = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}
_CARPET_PTS = [0, -1, 2, 4, 6, 10, 15, 21]


class PlayerAgent:
    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.rat_belief = None
        if transition_matrix is not None:
            from .rat_belief import RatBelief

            self.rat_belief = RatBelief(transition_matrix)

    def play(
        self,
        board: Board,
        sensor_data: Tuple,
        time_left: Callable,
    ) -> Move:
        noise, est_distance = sensor_data

        if self.rat_belief is not None:
            self.rat_belief.update(board, int(noise), int(est_distance))

        my_loc = board.player_worker.get_location()
        opp_loc = board.opponent_worker.get_location()
        turns_left = max(1, board.player_worker.turns_left)

        valid_moves = board.get_valid_moves(enemy=False, exclude_search=False)

        # Split into carpets, plains, and searches only — never prime
        carpets = [m for m in valid_moves if m.move_type == MoveType.CARPET]
        plains = [m for m in valid_moves if m.move_type == MoveType.PLAIN]

        # ── 1. Take any profitable carpet roll ──────────────────────────
        best_carpet = None
        best_carpet_pts = 0
        for mv in carpets:
            pts = _CARPET_PTS[min(mv.roll_length, 7)]
            if pts > best_carpet_pts:
                best_carpet_pts = pts
                best_carpet = mv

        if best_carpet is not None and best_carpet_pts >= 2:
            return best_carpet

        # ── 2. Search for the rat when confident ────────────────────────
        if self.rat_belief is not None:
            best_xy, search_ev = self.rat_belief.best_search_target()
            prob = self.rat_belief.best_prob

            should_search = False
            if turns_left <= 4 and prob >= 0.28:
                should_search = True
            elif turns_left <= 8 and prob >= 0.32:
                should_search = True
            elif turns_left <= 16 and prob >= 0.36:
                should_search = True
            elif prob >= 0.42:
                should_search = True

            if should_search:
                return Move.search(best_xy)

        # ── 3. Move toward the nearest primed cell to set up a carpet ───
        # Find the closest primed cell we could carpet from
        target = _best_carpet_target(my_loc, board._primed_mask)

        if target is not None:
            best_plain = _best_plain_toward(my_loc, target, plains)
            if best_plain is not None:
                return best_plain

        # ── 4. No primed cells to target — follow the opponent ──────────
        best_plain = _best_plain_toward(my_loc, opp_loc, plains)
        if best_plain is not None:
            return best_plain

        # Fallback
        if plains:
            return plains[0]
        if carpets:
            return carpets[0]
        return Move.search((0, 0))

    def commentate(self):
        if self.rat_belief is not None:
            best_pos = self.rat_belief.best_guess()
            best_prob = self.rat_belief.best_prob
            return f"Follow agent | Rat: {best_pos} ({best_prob:.1%})"
        return "Follow agent"


def _best_plain_toward(my_loc, target, plains):
    """Pick the plain move that gets us closest to target."""
    best = None
    best_dist = abs(my_loc[0] - target[0]) + abs(my_loc[1] - target[1])
    for mv in plains:
        new_loc = loc_after_direction(my_loc, mv.direction)
        if not (0 <= new_loc[0] < BOARD_SIZE and 0 <= new_loc[1] < BOARD_SIZE):
            continue
        dist = abs(new_loc[0] - target[0]) + abs(new_loc[1] - target[1])
        if dist < best_dist:
            best_dist = dist
            best = mv
    return best


def _best_carpet_target(my_loc, primed_mask):
    """
    Find the primed cell closest to us that is part of a run of length >= 2.
    We want to walk next to a run and then carpet it.
    To carpet a run, we need to be adjacent to one end of it.
    """
    if primed_mask == 0:
        return None

    best_loc = None
    best_dist = 999

    # Scan for horizontal primed runs of length >= 2
    for y in range(BOARD_SIZE):
        x = 0
        while x < BOARD_SIZE:
            if primed_mask & (1 << (y * 8 + x)):
                # Found start of a run, measure its length
                run_start = x
                while x < BOARD_SIZE and (primed_mask & (1 << (y * 8 + x))):
                    x += 1
                run_end = x - 1
                run_len = x - run_start
                if run_len >= 2:
                    # We need to be adjacent to an end to carpet this run.
                    # Check both endpoints: one cell before run_start or one
                    # cell after run_end, on the same row.
                    for tx in (run_start - 1, run_end + 1):
                        if 0 <= tx < BOARD_SIZE:
                            d = abs(my_loc[0] - tx) + abs(my_loc[1] - y)
                            if d < best_dist:
                                best_dist = d
                                best_loc = (tx, y)
            else:
                x += 1

    # Scan for vertical primed runs of length >= 2
    for x in range(BOARD_SIZE):
        y = 0
        while y < BOARD_SIZE:
            if primed_mask & (1 << (y * 8 + x)):
                run_start = y
                while y < BOARD_SIZE and (primed_mask & (1 << (y * 8 + x))):
                    y += 1
                run_end = y - 1
                run_len = y - run_start
                if run_len >= 2:
                    for ty in (run_start - 1, run_end + 1):
                        if 0 <= ty < BOARD_SIZE:
                            d = abs(my_loc[0] - x) + abs(my_loc[1] - ty)
                            if d < best_dist:
                                best_dist = d
                                best_loc = (x, ty)
            else:
                y += 1

    return best_loc
