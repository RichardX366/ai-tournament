"""
Monte Carlo Tree Search (MCTS) implementation for the ai-tournament game.

Usage in an agent:

    from helpers import MCTS

    class PlayerAgent:
        def __init__(self, board, transition_matrix=None, time_left=None):
            self.mcts = MCTS(
                time_limit=0.5,       # seconds per move
                exploration=1.41,     # UCB1 exploration constant
                rollout_depth=20,     # max random rollout depth
            )

        def play(self, board, sensor_data, time_left):
            return self.mcts.search(board)
"""

import math
import random
import time
from typing import List, Optional

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'engine'))

from game.board import Board
from game.move import Move
from game.enums import MoveType, Result, CARPET_POINTS_TABLE


class MCTSNode:
    """A node in the MCTS tree."""

    __slots__ = (
        'board', 'move', 'parent', 'children',
        'visits', 'total_value', 'untried_moves', 'is_maximizing',
    )

    def __init__(
        self,
        board: Board,
        move: Optional[Move] = None,
        parent: Optional['MCTSNode'] = None,
        is_maximizing: bool = True,
    ):
        self.board = board
        self.move = move          # the move that led to this node
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.total_value = 0.0
        self.is_maximizing = is_maximizing
        self.untried_moves: Optional[List[Move]] = None

    def expand_one(self) -> 'MCTSNode':
        """Expand one untried move and return the new child node."""
        if self.untried_moves is None:
            self.untried_moves = self.board.get_valid_moves(exclude_search=True)
            random.shuffle(self.untried_moves)

        move = self.untried_moves.pop()
        child_board = self.board.get_copy()
        child_board.apply_move(move, check_ok=False)

        if not child_board.is_game_over():
            child_board.reverse_perspective()

        child = MCTSNode(
            board=child_board,
            move=move,
            parent=self,
            is_maximizing=not self.is_maximizing,
        )
        self.children.append(child)
        return child

    def is_fully_expanded(self) -> bool:
        if self.untried_moves is None:
            self.untried_moves = self.board.get_valid_moves(exclude_search=True)
            random.shuffle(self.untried_moves)
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        return self.board.is_game_over()

    def best_child(self, exploration: float) -> 'MCTSNode':
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
        time_limit: seconds to spend searching per move
        max_iterations: hard cap on iterations (0 = unlimited, use time_limit)
        exploration: UCB1 exploration constant (sqrt(2) ~ 1.41 is standard)
        rollout_depth: max depth for random rollout before evaluating
        eval_fn: optional custom evaluation function(board) -> float in [-1, 1]
                 from the perspective of the *current* player_worker.
                 If None, uses the built-in heuristic.
    """

    def __init__(
        self,
        time_limit: float = 0.5,
        max_iterations: int = 0,
        exploration: float = 1.41,
        rollout_depth: int = 20,
        eval_fn=None,
    ):
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.exploration = exploration
        self.rollout_depth = rollout_depth
        self.eval_fn = eval_fn or self._default_eval

    def search(self, board: Board) -> Move:
        """
        Run MCTS from the given board state and return the best move.

        The board is NOT modified.
        """
        root = MCTSNode(board=board.get_copy(), is_maximizing=True)

        deadline = time.time() + self.time_limit
        iterations = 0

        while True:
            if time.time() >= deadline:
                break
            if self.max_iterations > 0 and iterations >= self.max_iterations:
                break

            # --- Selection ---
            node = root
            while not node.is_terminal() and node.is_fully_expanded():
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
            # Fallback: return a random valid move
            moves = board.get_valid_moves(exclude_search=True)
            return random.choice(moves) if moves else Move.plain(0)

        best = max(root.children, key=lambda c: c.visits)
        return best.move

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
        Rollout move selection policy. Biased toward higher-value moves
        to improve rollout quality while staying fast.
        """
        # Prefer carpet moves (high scoring) if available
        carpet_moves = [m for m in moves if m.move_type == MoveType.CARPET and m.roll_length >= 2]
        if carpet_moves:
            return max(carpet_moves, key=lambda m: CARPET_POINTS_TABLE[m.roll_length])

        # Prefer prime moves (set up future carpets)
        prime_moves = [m for m in moves if m.move_type == MoveType.PRIME]
        if prime_moves:
            return random.choice(prime_moves)

        return random.choice(moves)

    @staticmethod
    def _default_eval(board: Board) -> float:
        """
        Evaluate board from the perspective of the current player_worker.
        Returns a value in [-1, 1].

        Uses point differential normalized by a reasonable max score.
        """
        if board.is_game_over():
            winner = board.get_winner()
            if winner == Result.PLAYER:
                return 1.0
            elif winner == Result.ENEMY:
                return -1.0
            else:
                return 0.0

        my_points = board.player_worker.get_points()
        opp_points = board.opponent_worker.get_points()
        diff = my_points - opp_points

        # Count primed cells as potential future value for the current player.
        # Each primed cell near the player is a carpet opportunity.
        primed_count = bin(board._primed_mask).count('1')
        potential_bonus = primed_count * 0.3

        # Normalize: max realistic score diff is ~100
        score = (diff + potential_bonus) / 100.0
        return max(-1.0, min(1.0, score))
