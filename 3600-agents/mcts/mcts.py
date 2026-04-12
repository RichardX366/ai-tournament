"""
Monte Carlo Tree Search (MCTS) implementation for the ai-tournament game.

Usage in an agent:

    from .mcts import MCTS

    class PlayerAgent:
        def __init__(self, board, transition_matrix=None, time_left=None):
            self.mcts = MCTS(exploration=1.41, rollout_depth=20)

        def play(self, board, sensor_data, time_left):
            move, value = self.mcts.search(board, time_budget=0.5)
            return move
"""

import math
import random
import time
from typing import List, Optional, Tuple

from game.board import Board
from game.move import Move
from game.enums import MoveType, Direction, Result, BOARD_SIZE, CARPET_POINTS_TABLE


class MCTSNode:
    """A node in the MCTS tree."""

    __slots__ = (
        "board",
        "move",
        "parent",
        "children",
        "visits",
        "total_value",
        "untried_moves",
        "is_maximizing",
    )

    def __init__(
        self,
        board: Board,
        move: Optional[Move] = None,
        parent: Optional["MCTSNode"] = None,
        is_maximizing: bool = True,
    ):
        self.board = board
        self.move = move  # the move that led to this node
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.total_value = 0.0
        self.is_maximizing = is_maximizing
        self.untried_moves: Optional[List[Move]] = None

    def _init_untried(self):
        if self.untried_moves is None:
            self.untried_moves = self.board.get_valid_moves(exclude_search=True)
            random.shuffle(self.untried_moves)

    def expand_one(self) -> "MCTSNode":
        """Expand one untried move and return the new child node."""
        self._init_untried()
        move = self.untried_moves.pop()
        child_board = self.board.get_copy()
        child_board.apply_move(move, check_ok=False)

        # Only flip perspective and is_maximizing together — if the game
        # ended, player_worker is still the player who moved, so the
        # child must keep the same is_maximizing as the parent.
        child_is_max = self.is_maximizing
        if not child_board.is_game_over():
            child_board.reverse_perspective()
            child_is_max = not self.is_maximizing

        child = MCTSNode(
            board=child_board,
            move=move,
            parent=self,
            is_maximizing=child_is_max,
        )
        self.children.append(child)
        return child

    def is_fully_expanded(self) -> bool:
        self._init_untried()
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        return self.board.is_game_over()

    def best_child(self, exploration: float) -> "MCTSNode":
        """Select best child using UCB1."""
        log_parent = math.log(self.visits)

        best = None
        best_score = -math.inf

        for child in self.children:
            if child.visits == 0:
                return child
            exploit = child.total_value / child.visits
            explore = exploration * math.sqrt(log_parent / child.visits)
            # Maximizing player wants high value, minimizing wants low value
            if self.is_maximizing:
                score = exploit + explore
            else:
                score = -exploit + explore
            if score > best_score:
                best_score = score
                best = child

        return best

    def backpropagate(self, value: float):
        """Propagate value up to root."""
        node = self
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent


class MCTS:
    """
    Monte Carlo Tree Search.

    Parameters:
        exploration: UCB1 exploration constant (sqrt(2) ~ 1.41 is standard)
        rollout_depth: max depth for random rollout before evaluating
        eval_fn: optional custom evaluation function(board) -> float in [-1, 1]
                 from the perspective of the *current* player_worker.
                 If None, uses the built-in heuristic.
        rat_belief: optional RatBelief instance — if provided, the heuristic
                    uses rat position probabilities to bonus cells near the rat.
    """

    def __init__(
        self,
        exploration: float = 1.41,
        rollout_depth: int = 10,
        eval_fn=None,
        rat_belief=None,
    ):
        self.exploration = exploration
        self.rollout_depth = rollout_depth
        self.rat_belief = rat_belief
        self.eval_fn = eval_fn or self._make_eval(rat_belief)
        self._prev_root = None  # for tree reuse

    def set_rat_belief(self, rat_belief):
        """Update rat belief reference and rebuild eval fn."""
        self.rat_belief = rat_belief
        self.eval_fn = self._make_eval(rat_belief)

    def _make_eval(self, rat_belief):
        """Return an eval function closed over rat_belief."""
        def eval_fn(board):
            return MCTS._default_eval(board, rat_belief)
        return eval_fn

    def search(self, board: Board, time_budget: float) -> Tuple[Optional[Move], float]:
        """
        Run MCTS and return (best_move, average_value).

        average_value is from the root player's perspective in [-1, 1].
        The board is NOT modified. Reuses the subtree from the previous
        turn when possible.
        """
        root = self._try_reuse(board)
        if root is None:
            root = MCTSNode(board=board.get_copy(), is_maximizing=True)

        deadline = time.perf_counter() + time_budget
        iterations = 0
        _perf_counter = time.perf_counter  # local lookup avoids attr resolution

        while True:
            # Check time every 32 iterations to reduce syscall overhead
            if iterations & 31 == 0 and _perf_counter() >= deadline:
                break

            # --- Selection ---
            node = root
            while not node.is_terminal() and node.is_fully_expanded() and node.children:
                node = node.best_child(self.exploration)

            # --- Expansion ---
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand_one()

            # --- Simulation ---
            value = self._rollout(node.board, node.is_maximizing)

            # --- Backpropagation ---
            node.backpropagate(value)
            iterations += 1

        # Pick the most-visited child (most robust choice)
        if not root.children:
            self._prev_root = None
            moves = board.get_valid_moves(exclude_search=True)
            if moves:
                return random.choice(moves), 0.0
            return Move.plain(Direction.UP), 0.0

        best = max(root.children, key=lambda c: c.visits)
        avg_val = best.total_value / best.visits if best.visits > 0 else 0.0

        # Store the chosen child's subtree for reuse next turn.
        # Detach from parent to free the rest of the tree.
        best.parent = None
        self._prev_root = best

        return best.move, avg_val

    def _try_reuse(self, board: Board) -> Optional[MCTSNode]:
        """
        Try to find a reusable subtree from the previous turn's search.

        After our last move, the opponent moved. The board we receive has
        already been perspective-reversed (we are player_worker again).
        We look two levels deep in the previous tree:
          prev_root (our move) -> child (opponent's response) -> grandchild
        The grandchild matching the current board state becomes the new root.

        Falls back to None if no match is found — a fresh tree is built.
        """
        prev = self._prev_root
        self._prev_root = None
        if prev is None or not prev.children:
            return None

        # The prev_root was the child we chose last turn (opponent's perspective,
        # is_maximizing=False). Its children represent opponent's moves.
        # We need to find the grandchild matching the current board state.
        # Match by comparing board turn_count and worker positions.
        target_turn = board.turn_count
        my_pos = board.player_worker.get_location()
        opp_pos = board.opponent_worker.get_location()

        for opp_child in prev.children:
            # opp_child represents opponent's move — board is from our perspective
            # (after reverse_perspective), so is_maximizing=True
            if not opp_child.children:
                # Check if this node itself matches
                b = opp_child.board
                if (b.turn_count == target_turn and
                    b.player_worker.get_location() == my_pos and
                    b.opponent_worker.get_location() == opp_pos):
                    opp_child.parent = None
                    return opp_child
                continue

        # No match found — opponent made a move we didn't expand
        return None

    def _rollout(self, board: Board, is_maximizing: bool) -> float:
        """
        Simulate a random playout from the given state and return a value
        in [-1, 1] from the root player's perspective.
        """
        sim = board.get_copy()

        depth = 0
        current_is_max = is_maximizing

        while depth < self.rollout_depth and not sim.is_game_over():
            moves = sim.get_valid_moves(exclude_search=True)
            if not moves:
                break

            move = self._rollout_policy(moves, sim)
            sim.apply_move(move, check_ok=False)

            if not sim.is_game_over():
                sim.reverse_perspective()
                current_is_max = not current_is_max

            depth += 1

        # Evaluate from the perspective of whoever is currently player_worker.
        # If current_is_max == True, then player_worker is the root player,
        # so positive eval is good. If False, we negate.
        raw_eval = self.eval_fn(sim)
        return raw_eval if current_is_max else -raw_eval

    def _rollout_policy(self, moves: List[Move], board: Board) -> Move:
        """
        Single-pass rollout policy. Biased toward higher-value moves.
        Scans once: tracks best carpet, reservoir-samples one prime move.
        """
        best_carpet = None
        best_carpet_pts = -2  # worse than any real carpet
        reservoir_prime = None
        prime_count = 0

        _CARPET = MoveType.CARPET
        _PRIME = MoveType.PRIME
        _table = CARPET_POINTS_TABLE

        for m in moves:
            mt = m.move_type
            if mt == _CARPET and m.roll_length >= 2:
                pts = _table[m.roll_length]
                if pts > best_carpet_pts:
                    best_carpet_pts = pts
                    best_carpet = m
            elif mt == _PRIME:
                prime_count += 1
                # Reservoir sampling: each prime has equal 1/k chance of being chosen
                if random.randrange(prime_count) == 0:
                    reservoir_prime = m

        if best_carpet is not None:
            return best_carpet
        if reservoir_prime is not None:
            return reservoir_prime
        return random.choice(moves)

    @staticmethod
    def _default_eval(board: Board, rat_belief=None) -> float:
        """
        Evaluate board from the perspective of the current player_worker.
        Returns a value in [-1, 1].

        Components (all from player_worker's perspective):
          1. Point differential — actual scored points
          2. Carpet line potential — value of primed runs we can roll vs opponent
          3. Priming opportunity — how many open axes we're adjacent to
          4. Rat proximity bonus — being near the likely rat position is valuable
          5. Turns remaining differential
        """
        if board.is_game_over():
            winner = board.get_winner()
            if winner == Result.PLAYER:
                return 1.0
            elif winner == Result.ENEMY:
                return -1.0
            return 0.0

        my_pts = board.player_worker.get_points()
        opp_pts = board.opponent_worker.get_points()
        diff = my_pts - opp_pts

        # Proximity-weighted primed cell potential
        my_pos = board.player_worker.get_location()
        opp_pos = board.opponent_worker.get_location()
        primed_mask = board._primed_mask
        blocked_mask = board._blocked_mask
        carpet_mask  = board._carpet_mask

        # ── 1. Carpet line potential ──────────────────────────────────────────
        # For each axis (horizontal/vertical) scan primed runs and estimate
        # who is closer to rolling them. A run of length n is worth
        # CARPET_POINTS_TABLE[n] to whoever rolls it.
        my_carpet_ev  = 0.0
        opp_carpet_ev = 0.0

        for run_cells in _find_primed_runs(primed_mask, blocked_mask | carpet_mask):
            n = len(run_cells)
            if n < 1:
                continue
            pts = _table_lookup(n)

            # Closest endpoint of the run to each player
            # (worker needs to be adjacent to one end to roll)
            min_my_dist  = min(
                abs(my_pos[0]  - cx) + abs(my_pos[1]  - cy)
                for cx, cy in run_cells
            )
            min_opp_dist = min(
                abs(opp_pos[0] - cx) + abs(opp_pos[1] - cy)
                for cx, cy in run_cells
            )

            # Discount by distance — farther away means less likely to roll it
            # before the opponent or before the game ends
            my_share  = 1.0 / (1.0 + min_my_dist)
            opp_share = 1.0 / (1.0 + min_opp_dist)
            total     = my_share + opp_share

            my_carpet_ev  += pts * (my_share  / total)
            opp_carpet_ev += pts * (opp_share / total)

        carpet_advantage = my_carpet_ev - opp_carpet_ev

        # ── 2. Priming opportunity — open axes adjacent to us ─────────────────
        # An "open axis" is a direction from our current position where we
        # could start building a prime run. Worth less than carpet potential
        # but still signals board position quality.
        my_prime_axes  = _count_open_axes(my_pos,  primed_mask, blocked_mask, carpet_mask)
        opp_prime_axes = _count_open_axes(opp_pos, primed_mask, blocked_mask, carpet_mask)
        prime_advantage = my_prime_axes - opp_prime_axes

        # ── 3. Rat proximity bonus ────────────────────────────────────────────
        # If we're close to where the rat probably is, we could search soon.
        # Model this as: expected points if we searched right now minus
        # expected points if opponent were at our position (opportunity cost).
        rat_bonus = 0.0
        if rat_belief is not None:
            belief = rat_belief.belief
            # EV of searching from current position = 6*p_best - 2
            # But we also want to reward being physically close to the rat,
            # since the distance signal gets better near the rat.
            # Use a weighted sum of belief over cells near us.
            my_x, my_y = my_pos
            opp_x, opp_y = opp_pos

            my_rat_weight  = 0.0
            opp_rat_weight = 0.0

            # Iterate set bits of belief's top mass
            import jax.numpy as jnp
            belief_np = belief  # jnp array, indexable

            for idx in range(64):
                p = float(belief_np[idx])
                if p < 1e-4:
                    continue
                cx, cy = idx % 8, idx // 8
                my_dist  = abs(my_x  - cx) + abs(my_y  - cy)
                opp_dist = abs(opp_x - cx) + abs(opp_y - cy)
                # Closer to rat = more valuable (weight by inverse distance)
                my_rat_weight  += p / (1.0 + my_dist)
                opp_rat_weight += p / (1.0 + opp_dist)

            rat_bonus = (my_rat_weight - opp_rat_weight) * 4.0  # scale to point units

        # ── 4. Turns remaining ────────────────────────────────────────────────
        turn_diff = board.player_worker.turns_left - board.opponent_worker.turns_left

        # ── Combine ───────────────────────────────────────────────────────────
        # Weights tuned so that a ~5 point carpet advantage ≈ 0.25 on the scale,
        # keeping the output well within [-1, 1] for typical game states.
        score = (
            diff              * 1.0   # actual points are the ground truth
          + carpet_advantage  * 0.8   # unrealised carpet potential
          + prime_advantage   * 0.3   # positioning for future primes
          + rat_bonus         * 0.3   # rat proximity
          + turn_diff         * 0.2   # tempo
        ) / 40.0

        return max(-1.0, min(1.0, score))


# ── module-level helpers ──────────────────────────────────────────────────────

def _table_lookup(n: int) -> float:
    """Safe carpet points lookup, capped at length 7."""
    return float(CARPET_POINTS_TABLE.get(min(n, 7), 21))


def _count_primed_in_direction(pos, direction, primed_mask: int) -> int:
    """
    Count how many consecutive primed cells are in `direction` from `pos`.
    Used by the rollout policy to prefer priming toward existing runs.
    """
    from game.enums import loc_after_direction
    count = 0
    cur = pos
    for _ in range(BOARD_SIZE):
        cur = loc_after_direction(cur, direction)
        x, y = cur
        if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
            break
        bit = 1 << (y * BOARD_SIZE + x)
        if primed_mask & bit:
            count += 1
        else:
            break
    return count


def _find_primed_runs(primed_mask: int, obstacle_mask: int):
    """
    Yield lists of (x, y) tuples for each maximal horizontal or vertical
    run of primed cells unbroken by blocked/carpet cells.
    Only yields runs of length >= 1.
    """
    visited_h = set()
    visited_v = set()

    mask = primed_mask
    while mask:
        bit  = mask & (-mask)
        mask ^= bit
        idx  = bit.bit_length() - 1
        x, y = idx % BOARD_SIZE, idx // BOARD_SIZE

        # Horizontal run
        if idx not in visited_h:
            run = []
            # walk left to find start
            sx = x
            while sx > 0:
                left_bit = 1 << (y * BOARD_SIZE + sx - 1)
                if (primed_mask & left_bit) and not (obstacle_mask & left_bit):
                    sx -= 1
                else:
                    break
            # walk right collecting cells
            cx = sx
            while cx < BOARD_SIZE:
                cell_bit = 1 << (y * BOARD_SIZE + cx)
                if (primed_mask & cell_bit) and not (obstacle_mask & cell_bit):
                    run.append((cx, y))
                    visited_h.add(y * BOARD_SIZE + cx)
                    cx += 1
                else:
                    break
            if run:
                yield run

        # Vertical run
        if idx not in visited_v:
            run = []
            sy = y
            while sy > 0:
                up_bit = 1 << ((sy - 1) * BOARD_SIZE + x)
                if (primed_mask & up_bit) and not (obstacle_mask & up_bit):
                    sy -= 1
                else:
                    break
            cy = sy
            while cy < BOARD_SIZE:
                cell_bit = 1 << (cy * BOARD_SIZE + x)
                if (primed_mask & cell_bit) and not (obstacle_mask & cell_bit):
                    run.append((x, cy))
                    visited_v.add(cy * BOARD_SIZE + x)
                    cy += 1
                else:
                    break
            if run:
                yield run


def _count_open_axes(pos, primed_mask: int, blocked_mask: int, carpet_mask: int) -> int:
    """
    Count how many of the 4 cardinal directions from pos are 'open' —
    meaning the adjacent cell is SPACE (not primed, blocked, or carpet)
    so we could start or extend a prime run there.
    """
    from game.enums import loc_after_direction, Direction
    obstacle = primed_mask | blocked_mask | carpet_mask
    count = 0
    for d in Direction:
        nx, ny = loc_after_direction(pos, d)
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            bit = 1 << (ny * BOARD_SIZE + nx)
            if not (obstacle & bit):
                count += 1
    return count
