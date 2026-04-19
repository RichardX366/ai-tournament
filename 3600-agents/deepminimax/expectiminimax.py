import math
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

_rat_xy = None
_rat_prob = 0.0

# ── constants ─────────────────────────────────────────────────────────────────
_INF = float("inf")
_CARPET_PTS = [0, -1, 2, 4, 6, 10, 15, 21]
_CARPET_EVALUATIONS = [0.0, -1.0] + [
    (5 * _CARPET_PTS[L] - 1) / 6.0 for L in range(2, 8)
]

# Transposition table flag constants
_TT_EXACT = 0
_TT_LOWER = 1  # score is a lower bound (failed high / beta cutoff)
_TT_UPPER = 2  # score is an upper bound (failed low / alpha cutoff)
_TT_MAX_SIZE = 1 << 18  # 262144 entries

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


# ── transposition table helpers ───────────────────────────────────────────────


def _board_key(board):
    """Hashable key capturing the full mutable board state.

    Excludes _blocked_mask (constant per game) and _space_mask (derived).
    Includes everything that affects evaluation: tile masks, worker positions,
    scores, and turns remaining.
    """
    p = board.player_worker
    o = board.opponent_worker
    return (
        board._primed_mask,
        board._carpet_mask,
        p.position,
        o.position,
        p.points,
        o.points,
        p.turns_left,
        o.turns_left,
    )


def _move_key(move):
    """Compact hashable descriptor for a move (for TT best-move storage)."""
    return (move.move_type, move.direction, getattr(move, "roll_length", 0))


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
        if _rat_xy is not None and _rat_prob > 0.20:
            rx, ry = _rat_xy
            child_loc = child.player_worker.get_location()
            cx, cy = child_loc
            rat_dist = abs(cx - rx) + abs(cy - ry)
            score += 0.3 * _rat_prob * (1.0 / (rat_dist + 1))
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


def _best_carpet_for_sides(board, player_loc, opp_loc):
    """Scan primed runs of length >= 2, assign each exclusively to the closer
    player (Manhattan distance to outside endpoint; player wins ties).
    Returns (player_best, opp_best) with exponential distance decay."""
    primed = board._primed_mask
    blocked = board._blocked_mask
    if not primed:
        return 0.0, 0.0

    runs = []  # (ep1, ep2, val)  ep = (x, y) or None

    def add_run(x1, y1, x2, y2, val):
        inb1 = 0 <= x1 < 8 and 0 <= y1 < 8
        inb2 = 0 <= x2 < 8 and 0 <= y2 < 8
        b1 = not inb1 or bool(blocked & (1 << (y1 * 8 + x1)))
        b2 = not inb2 or bool(blocked & (1 << (y2 * 8 + x2)))
        if not b1 or not b2:
            runs.append((None if b1 else (x1, y1), None if b2 else (x2, y2), val))

    # Horizontal runs — endpoints are squares just outside the run
    for y in range(8):
        rs = rl = 0
        for x in range(9):
            has = x < 8 and bool(primed & (1 << (y * 8 + x)))
            if has:
                if rl == 0:
                    rs = x
                rl += 1
            elif rl >= 2:
                add_run(rs - 1, y, x, y, _CARPET_PTS[min(rl, 7)])
                rl = 0
            else:
                rl = 0

    # Vertical runs — endpoints are squares just outside the run
    for x in range(8):
        rs = rl = 0
        for y in range(9):
            has = y < 8 and bool(primed & (1 << (y * 8 + x)))
            if has:
                if rl == 0:
                    rs = y
                rl += 1
            elif rl >= 2:
                add_run(x, rs - 1, x, y, _CARPET_PTS[min(rl, 7)])
                rl = 0
            else:
                rl = 0

    if not runs:
        return 0.0, 0.0

    runs.sort(key=lambda r: r[2], reverse=True)

    px, py = player_loc
    ox, oy = opp_loc

    def dist_to_run(wx, wy, ep1, ep2):
        d = float("inf")
        if ep1 is not None:
            d = min(d, abs(ep1[0] - wx) + abs(ep1[1] - wy))
        if ep2 is not None:
            d = min(d, abs(ep2[0] - wx) + abs(ep2[1] - wy))
        return d

    player_best = 0.0
    opp_best = 0.0
    for ep1, ep2, val in runs:
        dp = dist_to_run(px, py, ep1, ep2)
        do = dist_to_run(ox, oy, ep1, ep2)
        if dp <= do:
            if player_best == 0.0:
                player_best = val * math.exp(-dp / 4)
        else:
            if opp_best == 0.0:
                opp_best = val * math.exp(-do / 4)
        if player_best > 0.0 and opp_best > 0.0:
            break

    return player_best, opp_best


def _evaluate(board: Board):
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

    score_diff = player.get_points() - opponent.get_points()

    tc = board.turn_count
    if tc > 77:
        return score_diff

    # Count primes for midgame factor
    player_moves = board.get_valid_moves(enemy=False, exclude_search=True)
    opponent_moves = board.get_valid_moves(enemy=True, exclude_search=True)
    player_prime_count = sum(1 for mv in player_moves if mv.move_type == MoveType.PRIME)
    opponent_prime_count = sum(1 for mv in opponent_moves if mv.move_type == MoveType.PRIME)

    # Primed-run ownership for best carpet
    player_best_carpet, opponent_best_carpet = _best_carpet_for_sides(
        board, player_loc, opponent_loc
    )

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

    # Ramp in turn 10→20, full 20→55, ramp out 55→65
    if tc < 10:
        midgame_multiplier = 0.0
    elif tc < 20:
        midgame_multiplier = (tc - 10) / 10.0
    elif tc < 55:
        midgame_multiplier = 1.0
    elif tc < 65:
        midgame_multiplier = (65 - tc) / 10.0
    else:
        midgame_multiplier = 0.0

    if midgame_multiplier > 0:
        player_lf, player_best_line = _line_freedom(
            board, player_loc, player_loc, opponent_loc
        )
        opponent_lf, opponent_best_line = _line_freedom(
            board, opponent_loc, player_loc, opponent_loc
        )
        midgame_factors = midgame_multiplier * (
            0.10 * (player_lf - opponent_lf)
            + 0.30 * (player_best_line - opponent_best_line)
            + 0.10 * (player_center - opponent_center)
            + 0.10 * (player_prime_count - opponent_prime_count)
            + 0.10 * (player_openness - opponent_openness)
            + 0.20 * (player_extension - opponent_extension)
        )
    else:
        midgame_factors = 0.0
    
    rat_pull = 0.0
    if _rat_xy is not None and _rat_prob > 0.20:
        rx, ry = _rat_xy
        p_dist = abs(player_x - rx) + abs(player_y - ry)
        o_dist = abs(opponent_x - rx) + abs(opponent_y - ry)
        rat_pull = 0.25 * _rat_prob * (o_dist - p_dist)
        # Positive when we're closer to the rat than opponent
    else:
        rat_pull = 0.0

    return (
        score_diff
        + 0.50 * (player_best_carpet - opponent_best_carpet)
        + midgame_factors
        + rat_pull
    )


def _line_freedom(board, loc, player_loc, opp_loc):
    """Score primable corridor potential: long SPACE lines through/near loc."""
    x, y = loc
    blocked = board._blocked_mask
    primed = board._primed_mask
    carpet = board._carpet_mask
    occupied = blocked | primed | carpet
    px, py = player_loc
    ox, oy = opp_loc

    total = 0.0
    best_runway = 0

    # Two axes: vertical (UP/DOWN) and horizontal (LEFT/RIGHT)
    axes = (((0, -1), (0, 1)), ((-1, 0), (1, 0)))

    for (dx1, dy1), (dx2, dy2) in axes:
        count1 = 0
        cx, cy = x + dx1, y + dy1
        while 0 <= cx < 8 and 0 <= cy < 8:
            bit = 1 << (cy * 8 + cx)
            if (occupied & bit) or (cx == px and cy == py) or (cx == ox and cy == oy):
                break
            if abs(cx - ox) + abs(cy - oy) < count1 + 1:
                break
            count1 += 1
            cx += dx1
            cy += dy1

        count2 = 0
        cx, cy = x + dx2, y + dy2
        while 0 <= cx < 8 and 0 <= cy < 8:
            bit = 1 << (cy * 8 + cx)
            if (occupied & bit) or (cx == px and cy == py) or (cx == ox and cy == oy):
                break
            if abs(cx - ox) + abs(cy - oy) < count2 + 1:
                break
            count2 += 1
            cx += dx2
            cy += dy2

        # Include self if primable
        bit_self = 1 << (y * 8 + x)
        self_ok = 0 if (occupied & bit_self) else 1
        runway = count1 + count2 + self_ok

        if runway > best_runway:
            best_runway = runway
        if runway >= 2:
            total += _CARPET_EVALUATIONS[min(runway, 7)]

    return total, best_runway


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
    """Negamax with PVS, transposition table, and iterative deepening."""

    def __init__(self, max_depth: int = 6):
        self._best_move: Optional[Move] = None
        self.max_depth = max_depth
        self._nodes = 0
        self._tt: dict = {}  # {board_key: (score, depth, flag, best_move_key)}
        self._tt_hits = 0
        self._rat_xy = None   # (x, y) best guess
        self._rat_prob = 0.0

    def search(self, board, rat_belief, time_budget=1.0):
        deadline = time.perf_counter() + time_budget
        self._deadline = deadline
        self._rat_belief = rat_belief
        self._nodes = 0
        self._tt_hits = 0

        # Keep TT across searches for cross-turn hits, but cap size
        if len(self._tt) > _TT_MAX_SIZE:
            self._tt.clear()

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
            _evaluate,
        )

        if not ordered:
            return move_moves[0], 0.0

        best_move = ordered[0]
        best_value = -_INF

        # Cap depth at remaining game turns so we find exact endgame scores
        turns_remaining = (
            board.player_worker.turns_left + board.opponent_worker.turns_left
        )
        effective_max = min(self.max_depth, max(1, turns_remaining - 1))

        best_moves = []

        global _rat_xy, _rat_prob
        if rat_belief is not None:
            _rat_xy = rat_belief.weighted_target()
            _rat_prob = rat_belief.best_prob
        else:
            _rat_xy, _rat_prob = None, 0.0

        # Iterative deepening with PVS
        for depth in range(1, effective_max + 1):
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
                best_moves.append((d_best, d_val))
                if d_best in ordered:
                    ordered.remove(d_best)
                    ordered.insert(0, d_best)
            if not done:
                break

        last_3: list[tuple[Move, float]] = (
            best_moves[-3:] if len(best_moves) >= 3 else best_moves
        )

        if last_3:
            summary = {}
            for idx, (mv, val) in enumerate(last_3):
                mk = _move_key(mv)
                if mk in summary:
                    count, _, _ = summary[mk]
                    summary[mk] = (count + 1, idx, (mv, val))
                else:
                    summary[mk] = (1, idx, (mv, val))

                _, _, (best_move, best_value) = max(
                    summary.values(),
                    key=lambda item: (item[0], item[1]),  # most common, else deepest
                )

        return best_move, best_value

    def _negamax(self, board, depth, alpha, beta):
        self._nodes += 1

        if depth <= 0 or board.is_game_over() or time.perf_counter() >= self._deadline:
            return _evaluate(board)

        # ── TT probe ────────────────────────────────────────────────────
        key = _board_key(board)
        tt_entry = self._tt.get(key)
        tt_best_mk = None

        if tt_entry is not None:
            tt_score, tt_depth, tt_flag, tt_mk = tt_entry
            if tt_depth >= depth:
                if tt_flag == _TT_EXACT:
                    self._tt_hits += 1
                    return tt_score
                if tt_flag == _TT_LOWER and tt_score >= beta:
                    self._tt_hits += 1
                    return tt_score
                if tt_flag == _TT_UPPER and tt_score <= alpha:
                    self._tt_hits += 1
                    return tt_score
            # Even if depth is insufficient for a cutoff, use the best
            # move for ordering (the most valuable part of the TT).
            tt_best_mk = tt_mk

        # ── Generate & order moves ──────────────────────────────────────
        moves = board.get_valid_moves(enemy=False, exclude_search=True)
        if not moves:
            return _evaluate(board)

        ordered = _order_moves_fast(board, moves)

        # Promote TT best move to front for better pruning
        if tt_best_mk is not None:
            for i in range(len(ordered)):
                if _move_key(ordered[i]) == tt_best_mk:
                    if i > 0:
                        ordered.insert(0, ordered.pop(i))
                    break

        # ── Search ──────────────────────────────────────────────────────
        orig_alpha = alpha
        best = -_INF
        best_mk = _move_key(ordered[0]) if ordered else None

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
                best_mk = _move_key(mv)
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break

        # ── TT store ────────────────────────────────────────────────────
        if best > -1e17:
            if best <= orig_alpha:
                flag = _TT_UPPER
            elif best >= beta:
                flag = _TT_LOWER
            else:
                flag = _TT_EXACT

            # Depth-preferred replacement: only overwrite if new search
            # is at least as deep as what's already stored.
            if tt_entry is None or depth >= tt_entry[1]:
                self._tt[key] = (best, depth, flag, best_mk)

        return best if best > -1e17 else _evaluate(board)
