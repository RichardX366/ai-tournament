import time
from typing import Optional, Tuple, List

import numpy as np

from game.board import Board
from game.move import Move
from game.enums import (
    MoveType,
    Direction,
    Result,
    BOARD_SIZE,
    CARPET_POINTS_TABLE,
    loc_after_direction,
)

# ── constants ─────────────────────────────────────────────────────────────────
_INF = float("inf")
_CARPET_PTS = [0, -1, 2, 4, 6, 10, 15, 21]
_CARPET_EVALUATIONS = [0.0, -1.0] + [
    (5 * _CARPET_PTS[L] - 1) / 6.0 for L in range(2, 8)
]

_DIRECTIONS = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
_DIRECTION_MOVEMENTS = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}
_OPPOSITE_DIRECTIONS = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}

# Pre-computed 64x64 Manhattan distance table
_INDEX_TO_LOCATION = [(i % BOARD_SIZE, i // BOARD_SIZE) for i in range(64)]
_DISTANCES = np.zeros((64, 64), dtype=np.float64)
for _wi in range(64):
    _wx, _wy = _INDEX_TO_LOCATION[_wi]
    for _ri in range(64):
        _rx, _ry = _INDEX_TO_LOCATION[_ri]
        _DISTANCES[_wi, _ri] = abs(_wx - _rx) + abs(_wy - _ry)

# Neighbor offsets for local openness (pre-computed)
_NEIGHBOR_OFFSETS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
_NEIGHBOR2 = []
for _dx, _dy in _NEIGHBOR_OFFSETS:
    nbr2 = []
    for _dx2, _dy2 in _NEIGHBOR_OFFSETS:
        nbr2.append((_dx + _dx2, _dy + _dy2))
    _NEIGHBOR2.append((_dx, _dy, nbr2))


# ── move ordering ─────────────────────────────────────────────────────────────


def _order_moves_full(board, moves, evaluate_fn):
    """Eval-based ordering at root for best pruning."""
    scored = []
    for move in moves:
        child = board.forecast_move(move, check_ok=True)
        if child is None:
            continue
        child.reverse_perspective()
        points = 0.0
        move_type = move.move_type
        if move_type == MoveType.PRIME:
            points = 0.5
        elif move_type == MoveType.CARPET:
            points = (
                _CARPET_EVALUATIONS[move.roll_length]
                if 0 <= move.roll_length < len(_CARPET_EVALUATIONS)
                else 0.0
            )
        score = -evaluate_fn(child) + 0.55 * points
        scored.append((score, move))
    scored.sort(key=lambda t: t[0], reverse=True)
    ordered = [move for _, move in scored]
    if len(ordered) > 16:
        carpets = [m for m in ordered if m.move_type == MoveType.CARPET][:6]
        primes = [m for m in ordered if m.move_type == MoveType.PRIME][:6]
        plains = [m for m in ordered if m.move_type == MoveType.PLAIN][:4]
        merged = []
        seen = set()
        for group in (carpets, primes, plains, ordered):
            for m in group:
                key = (m.move_type, m.direction, getattr(m, "roll_length", 0))
                if key not in seen:
                    merged.append(m)
                    seen.add(key)
                if len(merged) >= 16:
                    return merged
        return merged
    return ordered


def _order_moves_fast(board, moves):
    """Cheap ordering for internal nodes."""
    position = board.player_worker.get_location()
    pos_x, pos_y = position
    primed_mask = board._primed_mask
    scored = []
    for move in moves:
        move_type = move.move_type
        if move_type == MoveType.CARPET:
            roll_length = move.roll_length
            score = (
                _CARPET_EVALUATIONS[roll_length]
                if 0 <= roll_length < len(_CARPET_EVALUATIONS)
                else 0.0
            ) * 3
        elif move_type == MoveType.PRIME:
            score = 1.0
            opposite_direction = _OPPOSITE_DIRECTIONS[move.direction]
            opposite_dx, opposite_dy = _DIRECTION_MOVEMENTS[opposite_direction]
            behind_x, behind_y = pos_x + opposite_dx, pos_y + opposite_dy
            if (
                0 <= behind_x < 8
                and 0 <= behind_y < 8
                and (primed_mask & (1 << (behind_y * 8 + behind_x)))
            ):
                score += 4
        else:
            score = 0.0
        scored.append((score, move))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [move for _, move in scored]


# ── static evaluation ─────────────────────────────────────────────────────────


def _evaluate(board, rat_belief):
    """Static evaluation from perspective of board.player_worker."""
    if board.is_game_over():
        w = board.get_winner()
        if w == Result.PLAYER:
            return 999.0
        if w == Result.ENEMY:
            return -999.0
        return 0.0

    player = board.player_worker
    opponent = board.opponent_worker
    player_loc = player.get_location()
    opponent_loc = opponent.get_location()

    score_diff = float(player.get_points() - opponent.get_points())
    turn_diff = float(player.turns_left - opponent.turns_left)

    player_moves = board.get_valid_moves(enemy=False, exclude_search=True)
    opponent_moves = board.get_valid_moves(enemy=True, exclude_search=True)

    # Inline _tact for both sides to avoid function call overhead
    player_best_carpet = 0
    player_carpet_sum = 0
    player_prime_count = 0
    for mv in player_moves:
        mt = mv.move_type
        if mt == MoveType.CARPET:
            rl = mv.roll_length
            pts = _CARPET_EVALUATIONS[rl] if 0 <= rl < len(_CARPET_EVALUATIONS) else 0.0
            if pts > player_best_carpet:
                player_best_carpet = pts
            if pts > 0:
                player_carpet_sum += pts
        elif mt == MoveType.PRIME:
            player_prime_count += 1

    opponent_best_carpet = 0
    opponent_carpet_sum = 0
    opponent_prime_count = 0
    for mv in opponent_moves:
        mt = mv.move_type
        if mt == MoveType.CARPET:
            rl = mv.roll_length
            pts = _CARPET_EVALUATIONS[rl] if 0 <= rl < len(_CARPET_EVALUATIONS) else 0.0
            if pts > opponent_best_carpet:
                opponent_best_carpet = pts
            if pts > 0:
                opponent_carpet_sum += pts
        elif mt == MoveType.PRIME:
            opponent_prime_count += 1

    # Local openness — bitmask-based for speed
    blocked = board._blocked_mask
    player_x, player_y = player_loc
    opponent_x, opponent_y = opponent_loc
    player_openness = 0
    opponent_openness = 0
    for dx, dy, nbr2 in _NEIGHBOR2:
        nx, ny = player_x + dx, player_y + dy
        if 0 <= nx < 8 and 0 <= ny < 8 and not (blocked & (1 << (ny * 8 + nx))):
            player_openness += 2
            for dx2, dy2 in nbr2:
                nx2, ny2 = player_x + dx2, player_y + dy2
                if (
                    0 <= nx2 < 8
                    and 0 <= ny2 < 8
                    and not (blocked & (1 << (ny2 * 8 + nx2)))
                ):
                    player_openness += 1
        nx, ny = opponent_x + dx, opponent_y + dy
        if 0 <= nx < 8 and 0 <= ny < 8 and not (blocked & (1 << (ny * 8 + nx))):
            opponent_openness += 2
            for dx2, dy2 in nbr2:
                nx2, ny2 = opponent_x + dx2, opponent_y + dy2
                if (
                    0 <= nx2 < 8
                    and 0 <= ny2 < 8
                    and not (blocked & (1 << (ny2 * 8 + nx2)))
                ):
                    opponent_openness += 1

    player_center = 7.0 - (abs(player_x - 3.5) + abs(player_y - 3.5))
    opponent_center = 7.0 - (abs(opponent_x - 3.5) + abs(opponent_y - 3.5))

    player_extension = _ext(board, player_loc, player_loc, opponent_loc)
    opponent_extension = _ext(board, opponent_loc, player_loc, opponent_loc)

    v = (
        3.0 * score_diff
        + 0.12 * turn_diff
        + 0.28 * (len(player_moves) - len(opponent_moves))
        + 0.95 * (player_best_carpet - opponent_best_carpet)
        + 0.18 * (player_carpet_sum - opponent_carpet_sum)
        + 0.35 * (player_prime_count - opponent_prime_count)
        + 0.22 * (player_openness - opponent_openness)
        + 0.14 * (player_center - opponent_center)
        + 0.45 * (player_extension - opponent_extension)
    )

    # Rat proximity — numpy vectorized
    if rat_belief is not None:
        belief = rat_belief.belief
        player_idx = player_y * 8 + player_x
        opponent_idx = opponent_y * 8 + opponent_x
        v += 0.10 * float(
            np.dot(
                belief,
                _DISTANCES[opponent_idx].astype(np.float64)
                - _DISTANCES[player_idx].astype(np.float64),
            )
        )

    return v


def _ext(board, loc, player_loc, opp_loc):
    """Line extension value: future carpet potential from position."""
    x, y = loc
    bit = 1 << (y * 8 + x)
    primed = board._primed_mask
    carpet = board._carpet_mask
    blocked = board._blocked_mask
    if (primed | carpet | blocked) & bit:
        return 0.0
    best = 0.0
    for direction in _DIRECTIONS:
        dx, dy = _DIRECTION_MOVEMENTS[direction]
        next_x, next_y = x + dx, y + dy
        if not (0 <= next_x < 8 and 0 <= next_y < 8):
            continue
        next_bit = 1 << (next_y * 8 + next_x)
        if (blocked | primed) & next_bit:
            continue
        if (next_x, next_y) == player_loc or (next_x, next_y) == opp_loc:
            continue
        opposite_dir = _OPPOSITE_DIRECTIONS[direction]
        opp_dx, opp_dy = _DIRECTION_MOVEMENTS[opposite_dir]
        primed_behind = 0
        behind_x, behind_y = x + opp_dx, y + opp_dy
        while (
            0 <= behind_x < 8
            and 0 <= behind_y < 8
            and (primed & (1 << (behind_y * 8 + behind_x)))
        ):
            primed_behind += 1
            behind_x += opp_dx
            behind_y += opp_dy
        available_ahead = 0
        ahead_x, ahead_y = next_x + dx, next_y + dy
        space = board._space_mask
        while 0 <= ahead_x < 8 and 0 <= ahead_y < 8:
            ahead_bit = 1 << (ahead_y * 8 + ahead_x)
            if (space & ahead_bit) and not ((primed | carpet | blocked) & ahead_bit):
                available_ahead += 1
                ahead_x += dx
                ahead_y += dy
            else:
                break
        if primed_behind > 0:
            val = (
                _CARPET_EVALUATIONS[min(primed_behind + 1, 7)] * 0.5
                + available_ahead * 0.15
            )
        else:
            val = available_ahead * 0.2
        if val > best:
            best = val
    return best


# ── main search class ─────────────────────────────────────────────────────────


class Expectiminimax:
    """Negamax with PVS, width-limited alpha-beta, and iterative deepening."""

    def __init__(self, max_depth: int = 6):
        self._best_move: Optional[Move] = None
        self.max_depth = max_depth
        self._nodes = 0

    def search(self, board, rat_belief, time_left_fn, time_budget=1.0):
        deadline = time.perf_counter() + time_budget
        self._deadline = deadline
        self._rat_belief = rat_belief
        self._nodes = 0

        moves = board.get_valid_moves(exclude_search=False)
        if not moves:
            return Move.plain(Direction.UP), 0.0

        search_moves = [m for m in moves if m.move_type == MoveType.SEARCH]
        move_moves = [m for m in moves if m.move_type != MoveType.SEARCH]

        if not move_moves:
            return (search_moves[0] if search_moves else Move.plain(Direction.UP)), 0.0

        # Full eval-based ordering at root
        ordered = _order_moves_full(
            board,
            move_moves,
            lambda b: _evaluate(b, rat_belief),
        )

        if not ordered:
            return move_moves[0], 0.0

        best_move = ordered[0]
        best_value = -_INF

        # Iterative deepening with PVS
        for depth in range(1, self.max_depth + 1):
            if time.perf_counter() >= deadline:
                break

            alpha = -_INF
            beta = _INF
            d_best = ordered[0]
            d_val = -_INF
            done = True

            for i, mv in enumerate(ordered):
                if time.perf_counter() >= deadline:
                    done = False
                    break

                child = board.forecast_move(mv, check_ok=True)
                if child is None:
                    continue
                child.reverse_perspective()

                if i == 0:
                    val = -self._negamax(child, depth - 1, -beta, -alpha)
                else:
                    val = -self._negamax(child, depth - 1, -alpha - 0.01, -alpha)
                    if val > alpha and val < beta:
                        val = -self._negamax(child, depth - 1, -beta, -alpha)

                if val > d_val:
                    d_val = val
                    d_best = mv
                if val > alpha:
                    alpha = val

            if d_val > -1e17:
                best_value = d_val
                best_move = d_best
                if d_best in ordered:
                    ordered.remove(d_best)
                    ordered.insert(0, d_best)
            if not done:
                break
        # Decide: movement vs search
        if rat_belief is not None and search_moves:
            best_search_xy, search_ev = rat_belief.best_search_target()
            turns_left = board.player_worker.turns_left

            if turns_left > 25:
                threshold = 2.0
            elif turns_left > 15:
                threshold = 1.0
            elif turns_left > 5:
                threshold = 0.5
            else:
                threshold = 0.0

            if search_ev > threshold:
                search_value = search_ev / 42.0
                if search_value > best_value + 0.05:
                    return Move.search(best_search_xy), search_value

        return best_move, best_value

    def _negamax(self, board, depth, alpha, beta):
        self._nodes += 1

        if depth <= 0 or board.is_game_over() or time.perf_counter() >= self._deadline:
            return _evaluate(board, self._rat_belief)

        moves = board.get_valid_moves(enemy=False, exclude_search=True)
        if not moves:
            return _evaluate(board, self._rat_belief)

        ordered = _order_moves_fast(board, moves)
        # width = 10  # width doesn't seem to have an effect because there are only around 8 moves on average
        # ordered = ordered[:width]

        best = -_INF
        for i, mv in enumerate(ordered):
            if time.perf_counter() >= self._deadline:
                break

            child = board.forecast_move(mv, check_ok=True)
            if child is None:
                continue
            child.reverse_perspective()

            if i == 0:
                val = -self._negamax(child, depth - 1, -beta, -alpha)
            else:
                val = -self._negamax(child, depth - 1, -alpha - 0.01, -alpha)
                if val > alpha and val < beta:
                    val = -self._negamax(child, depth - 1, -beta, -alpha)

            if val > best:
                best = val
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break

        return best if best > -1e17 else _evaluate(board, self._rat_belief)
