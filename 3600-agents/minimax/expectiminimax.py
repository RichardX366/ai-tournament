"""
expectiminimax.py — Expectiminimax search with alpha-beta pruning.

The game tree has three node types:
  - MAX nodes  : our turn — pick the move with the highest value
  - MIN nodes  : opponent's turn — pick the move with the lowest value
  - CHANCE nodes: rat moves before each turn — take weighted average
                  over rat positions using the current belief distribution

The "expecti" part is lightweight: rather than branching on every possible
rat position, we bake the rat belief into the static evaluation function
(as a rat-proximity bonus). This keeps branching factor manageable while
still making rat-aware decisions.

Alpha-beta pruning cuts branches that cannot affect the final decision,
roughly doubling the effective search depth vs plain minimax.

Usage:
    from .expectiminimax import Expectiminimax

    searcher = Expectiminimax(max_depth=4)
    best_move, value = searcher.search(board, rat_belief, time_left_fn)
"""

import time
from typing import Optional, Tuple, List

from game.board import Board
from game.move import Move
from game.enums import (
    MoveType, Direction, Result,
    BOARD_SIZE, CARPET_POINTS_TABLE,
    loc_after_direction,
)

# ── constants ─────────────────────────────────────────────────────────────────
_INF = float("inf")

# Carpet points table as a plain list for O(1) lookup (index = roll length)
_CARPET_PTS = [0, -1, 2, 4, 6, 10, 15, 21]  # index 0 unused


# ── move ordering ─────────────────────────────────────────────────────────────

def _move_score(move: Move, board: Board) -> float:
    """
    Score a move for ordering purposes (higher = search first).
    Good ordering is critical for alpha-beta efficiency.
    """
    mt = move.move_type
    if mt == MoveType.CARPET:
        # High-value carpet rolls first
        return 100.0 + _CARPET_PTS[min(move.roll_length, 7)]
    if mt == MoveType.PRIME:
        # Primes that extend existing runs are better
        pos = board.player_worker.get_location()
        count = _primed_in_direction(pos, move.direction, board._primed_mask)
        return 10.0 + count
    if mt == MoveType.SEARCH:
        return 50.0   # searches handled separately
    return 0.0        # plain moves last


def _primed_in_direction(pos, direction, primed_mask: int) -> int:
    """Count consecutive primed cells from pos in direction."""
    count = 0
    cur = pos
    for _ in range(BOARD_SIZE):
        cur = loc_after_direction(cur, direction)
        x, y = cur
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            break
        if primed_mask & (1 << (y * BOARD_SIZE + x)):
            count += 1
        else:
            break
    return count


# ── static evaluation ─────────────────────────────────────────────────────────

def _evaluate(board: Board, rat_belief) -> float:
    """
    Static evaluation from the perspective of board.player_worker.
    Returns a float in [-1, 1].

    Components:
      1. Point differential
      2. Carpet line potential (primed runs weighted by proximity)
      3. Future priming potential (open axes near each player)
      4. Rat proximity (expected value of being near the rat)
    """
    if board.is_game_over():
        w = board.get_winner()
        if w == Result.PLAYER: return  1.0
        if w == Result.ENEMY:  return -1.0
        return 0.0

    my_pos  = board.player_worker.get_location()
    opp_pos = board.opponent_worker.get_location()
    my_pts  = board.player_worker.get_points()
    opp_pts = board.opponent_worker.get_points()

    primed_mask  = board._primed_mask
    blocked_mask = board._blocked_mask
    carpet_mask  = board._carpet_mask
    obstacle     = blocked_mask | carpet_mask

    # ── 1. Point differential ─────────────────────────────────────────────────
    diff = float(my_pts - opp_pts)

    # ── 2. Carpet line potential ──────────────────────────────────────────────
    # For each primed run of length >= 2, estimate who is more likely to roll
    # it based on distance to the nearest endpoint.
    my_carpet_ev  = 0.0
    opp_carpet_ev = 0.0

    for run in _iter_primed_runs(primed_mask, obstacle):
        n   = len(run)
        pts = float(_CARPET_PTS[min(n, 7)])

        my_dist  = min(abs(my_pos[0]  - cx) + abs(my_pos[1]  - cy) for cx, cy in run)
        opp_dist = min(abs(opp_pos[0] - cx) + abs(opp_pos[1] - cy) for cx, cy in run)

        my_w  = 1.0 / (1.0 + my_dist)
        opp_w = 1.0 / (1.0 + opp_dist)
        total = my_w + opp_w

        my_carpet_ev  += pts * my_w  / total
        opp_carpet_ev += pts * opp_w / total

    carpet_adv = my_carpet_ev - opp_carpet_ev

    # ── 3. Future priming potential ───────────────────────────────────────────
    # Count open axes (directions with clear space to build a run).
    # Also reward being adjacent to an existing primed run end —
    # that's a free carpet roll waiting to happen.
    my_axes  = _open_axes(my_pos,  primed_mask, obstacle)
    opp_axes = _open_axes(opp_pos, primed_mask, obstacle)

    my_ext  = _run_extension_value(my_pos,  primed_mask, obstacle)
    opp_ext = _run_extension_value(opp_pos, primed_mask, obstacle)

    position_adv = (my_axes - opp_axes) * 0.5 + (my_ext - opp_ext)

    # ── 4. Rat proximity ──────────────────────────────────────────────────────
    rat_adv = 0.0
    if rat_belief is not None:
        belief = rat_belief.belief
        mx, my = my_pos
        ox, oy = opp_pos
        my_rat  = 0.0
        opp_rat = 0.0
        for idx in range(64):
            p = float(belief[idx])
            if p < 5e-4:
                continue
            cx, cy = idx % 8, idx // 8
            my_rat  += p / (1.0 + abs(mx - cx) + abs(my - cy))
            opp_rat += p / (1.0 + abs(ox - cx) + abs(oy - cy))
        rat_adv = (my_rat - opp_rat) * 4.0

    # ── Combine ───────────────────────────────────────────────────────────────
    raw = (
        diff         * 1.0
      + carpet_adv   * 0.9
      + position_adv * 0.3
      + rat_adv      * 0.25
    ) / 42.0

    return max(-1.0, min(1.0, raw))


def _iter_primed_runs(primed_mask: int, obstacle_mask: int):
    """Yield (x,y) lists for each maximal primed run of length >= 2."""
    visited_h: set = set()
    visited_v: set = set()
    mask = primed_mask
    while mask:
        bit  = mask & (-mask)
        mask ^= bit
        idx  = bit.bit_length() - 1
        x, y = idx % BOARD_SIZE, idx // BOARD_SIZE

        if idx not in visited_h:
            run = []
            sx  = x
            while sx > 0 and (primed_mask & (1 << (y * BOARD_SIZE + sx - 1))) and \
                  not (obstacle_mask & (1 << (y * BOARD_SIZE + sx - 1))):
                sx -= 1
            cx = sx
            while cx < BOARD_SIZE:
                b = 1 << (y * BOARD_SIZE + cx)
                if (primed_mask & b) and not (obstacle_mask & b):
                    run.append((cx, y))
                    visited_h.add(y * BOARD_SIZE + cx)
                    cx += 1
                else:
                    break
            if len(run) >= 2:
                yield run

        if idx not in visited_v:
            run = []
            sy  = y
            while sy > 0 and (primed_mask & (1 << ((sy - 1) * BOARD_SIZE + x))) and \
                  not (obstacle_mask & (1 << ((sy - 1) * BOARD_SIZE + x))):
                sy -= 1
            cy = sy
            while cy < BOARD_SIZE:
                b = 1 << (cy * BOARD_SIZE + x)
                if (primed_mask & b) and not (obstacle_mask & b):
                    run.append((x, cy))
                    visited_v.add(cy * BOARD_SIZE + x)
                    cy += 1
                else:
                    break
            if len(run) >= 2:
                yield run


def _open_axes(pos, primed_mask: int, obstacle_mask: int) -> int:
    """Count cardinal directions from pos that are free (SPACE, not obstructed)."""
    count = 0
    for d in Direction:
        nx, ny = loc_after_direction(pos, d)
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            b = 1 << (ny * BOARD_SIZE + nx)
            if not ((primed_mask | obstacle_mask) & b):
                count += 1
    return count


def _run_extension_value(pos, primed_mask: int, obstacle_mask: int) -> float:
    """
    How much carpet-roll value is available by stepping adjacent to existing runs.
    For each direction, count how many primed cells are lined up — if you prime
    in that direction you could roll all of them next turn.
    """
    total = 0.0
    for d in Direction:
        n = _primed_in_direction(pos, d, primed_mask)
        if n >= 2:
            total += float(_CARPET_PTS[min(n, 7)])
    return total


# ── main search class ─────────────────────────────────────────────────────────

class Expectiminimax:
    """
    Expectiminimax search with alpha-beta pruning and iterative deepening.

    The tree alternates MAX (our turn) and MIN (opponent's turn) nodes.
    Rat belief is incorporated in the static eval rather than as explicit
    chance nodes — this keeps the branching factor low while still making
    rat-aware decisions.

    Iterative deepening lets us use all available time: we search depth 1,
    then depth 2, etc., always keeping the best answer from the last
    completed depth. If time runs out mid-search we fall back gracefully.
    """

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self._best_move: Optional[Move] = None
        self._nodes = 0

    def search(
        self,
        board: Board,
        rat_belief,
        time_left_fn,
        time_budget: float = 1.0,
    ) -> Tuple[Optional[Move], float]:
        """
        Run iterative-deepening expectiminimax and return (best_move, value).

        Parameters
        ----------
        board        : current Board (not modified)
        rat_belief   : RatBelief instance (may be None)
        time_left_fn : callable returning seconds remaining
        time_budget  : seconds allocated for this search
        """
        deadline = time.perf_counter() + time_budget
        self._deadline = deadline
        self._timed_out = False

        moves = board.get_valid_moves(exclude_search=False)
        if not moves:
            return Move.plain(Direction.UP), 0.0

        # Separate search moves from movement moves
        search_moves = [m for m in moves if m.move_type == MoveType.SEARCH]
        move_moves   = [m for m in moves if m.move_type != MoveType.SEARCH]

        # Order movement moves for better alpha-beta cutoffs
        move_moves.sort(key=lambda m: _move_score(m, board), reverse=True)

        best_move  = move_moves[0] if move_moves else moves[0]
        best_value = -_INF

        # ── Iterative deepening ───────────────────────────────────────────────
        for depth in range(1, self.max_depth + 1):
            if time.perf_counter() >= deadline:
                break

            self._timed_out = False
            self._nodes = 0

            candidate_move, candidate_value = self._root_search(
                board, move_moves, rat_belief, depth, deadline
            )

            if not self._timed_out:
                # Only accept a completed search
                best_move  = candidate_move
                best_value = candidate_value
            else:
                break

        # ── Decide: best movement move vs best search move ────────────────────
        if rat_belief is not None and search_moves:
            best_search_xy, search_ev = rat_belief.best_search_target()
            turns_left = board.player_worker.turns_left

            # Adaptive threshold: tighter early, looser late.
            # EV = 6p - 2, so:
            #   threshold 2.0 → p > 0.667 (early game, very confident only)
            #   threshold 1.0 → p > 0.500 (mid game)
            #   threshold 0.0 → p > 0.333 (late game, any +EV)
            if turns_left > 25:
                threshold = 2.0
            elif turns_left > 15:
                threshold = 1.0
            elif turns_left > 5:
                threshold = 0.5
            else:
                threshold = 0.0

            if search_ev > threshold:
                # Only search if EV in points clearly beats the tree value
                # converted to the same scale (tree normalises by ~42)
                search_value = search_ev / 42.0
                # Require search to be meaningfully better, not just marginally
                if search_value > best_value + 0.05:
                    return Move.search(best_search_xy), search_value

        return best_move, best_value

    def _root_search(
        self,
        board: Board,
        moves: List[Move],
        rat_belief,
        depth: int,
        deadline: float,
    ) -> Tuple[Optional[Move], float]:
        """Search root node, returning (best_move, best_value)."""
        alpha = -_INF
        beta  =  _INF
        best_move  = moves[0] if moves else None
        best_value = -_INF

        for move in moves:
            if time.perf_counter() >= deadline:
                self._timed_out = True
                break

            child = board.get_copy()
            child.apply_move(move, check_ok=False)

            if not child.is_game_over():
                child.reverse_perspective()
                # After our move it's the opponent's turn (MIN node)
                value = -self._minimax(
                    child, depth - 1, -beta, -alpha,
                    rat_belief, deadline, is_max=False
                )
            else:
                value = _evaluate(child, rat_belief)

            if value > best_value:
                best_value = value
                best_move  = move

            alpha = max(alpha, value)

        return best_move, best_value

    def _minimax(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        rat_belief,
        deadline: float,
        is_max: bool,
    ) -> float:
        """
        Negamax with alpha-beta pruning.

        Uses negamax formulation: value is always from the perspective of
        board.player_worker, and we negate when recursing into the child
        (which has reversed perspective).

        is_max is tracked only for potential future use; negamax handles
        the max/min duality automatically through negation.
        """
        self._nodes += 1

        # Time check every 128 nodes to reduce perf_counter overhead
        if self._nodes & 127 == 0 and time.perf_counter() >= deadline:
            self._timed_out = True
            return 0.0

        if board.is_game_over() or depth == 0:
            return _evaluate(board, rat_belief)

        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
            return _evaluate(board, rat_belief)

        # Move ordering improves alpha-beta cutoffs significantly
        moves.sort(key=lambda m: _move_score(m, board), reverse=True)

        best = -_INF

        for move in moves:
            if self._timed_out:
                return 0.0

            child = board.get_copy()
            child.apply_move(move, check_ok=False)

            if not child.is_game_over():
                child.reverse_perspective()
                value = -self._minimax(
                    child, depth - 1, -beta, -alpha,
                    rat_belief, deadline, is_max=not is_max
                )
            else:
                value = _evaluate(child, rat_belief)

            best  = max(best, value)
            alpha = max(alpha, value)

            # Beta cutoff — prune remaining branches
            if alpha >= beta:
                break

        return best