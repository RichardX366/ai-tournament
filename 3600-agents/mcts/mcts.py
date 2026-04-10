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
    """

    def __init__(
        self,
        exploration: float = 1.41,
        rollout_depth: int = 10,
        eval_fn=None,
    ):
        self.exploration = exploration
        self.rollout_depth = rollout_depth
        self.eval_fn = eval_fn or self._default_eval
        self._prev_root = None  # for tree reuse

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
    def _default_eval(board: Board) -> float:
        """
        Evaluate board from the perspective of the current player_worker.
        Returns a value in [-1, 1].

        Components:
        - Point differential (dominant signal)
        - Proximity-weighted primed cells: primed cells near the player
          are carpet opportunities; primed cells near the opponent are threats.
        - Turns remaining advantage
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
        my_potential = 0.0
        opp_potential = 0.0

        # Iterate only over set bits
        mask = primed_mask
        while mask:
            # Isolate lowest set bit
            bit = mask & (-mask)
            mask ^= bit
            idx = bit.bit_length() - 1
            px = idx % BOARD_SIZE
            py = idx // BOARD_SIZE

            my_dist = abs(my_pos[0] - px) + abs(my_pos[1] - py)
            opp_dist = abs(opp_pos[0] - px) + abs(opp_pos[1] - py)

            # Closer cells are worth more (inverse distance, capped)
            # A primed cell right next to you is a carpet-2 opportunity (~2 pts)
            if my_dist <= 1:
                my_potential += 2.0
            elif my_dist <= 3:
                my_potential += 1.0
            else:
                my_potential += 0.3

            if opp_dist <= 1:
                opp_potential += 2.0
            elif opp_dist <= 3:
                opp_potential += 1.0
            else:
                opp_potential += 0.3

        # Net potential: our carpet opportunities minus opponent's
        potential = my_potential - opp_potential

        # Turns advantage (more turns = more opportunity to score)
        turn_diff = board.player_worker.turns_left - board.opponent_worker.turns_left

        score = (diff + potential * 0.4 + turn_diff * 0.3) / 80.0
        return max(-1.0, min(1.0, score))
