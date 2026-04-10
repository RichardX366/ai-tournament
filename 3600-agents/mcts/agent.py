"""
MCTS agent with Hidden Markov Model rat tracking.

Strategy:
- Uses MCTS to search the game tree for the best movement move.
- Maintains a probability distribution over rat positions using an HMM
  (transition matrix + noisy sensor observations).
- Decides between searching for the rat vs. making a movement move
  based on expected value comparison.
"""

from collections.abc import Callable
from typing import Tuple

from game.board import Board
from game.move import Move
from .mcts import MCTS
from .rat_belief import RatBelief


class PlayerAgent:
    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.mcts = MCTS(exploration=1.41, rollout_depth=10)
        self.rat_belief = None
        if transition_matrix is not None:
            self.rat_belief = RatBelief(transition_matrix)
        self.turn_number = 0

    def play(
        self,
        board: Board,
        sensor_data: Tuple,
        time_left: Callable,
    ) -> Move:
        self.turn_number += 1
        noise, est_distance = sensor_data

        # --- Update rat belief ---
        search_move = None
        search_ev = -float("inf")
        if self.rat_belief is not None:
            self.rat_belief.update(board, int(noise), int(est_distance))

            # Evaluate whether to search
            best_pos, best_ev = self.rat_belief.best_search_target()
            search_ev = best_ev
            if best_ev > 0:
                search_move = Move.search(best_pos)

        # --- Decide: search vs. move ---
        # search_ev is in points (EV = 6p - 2).
        # If search EV > 1.5 points, prefer searching — skip MCTS entirely.
        if search_move is not None and search_ev > 1.5:
            return search_move

        # --- Time management ---
        remaining = time_left()
        turns_left = board.player_worker.turns_left
        if turns_left <= 0:
            turns_left = 1

        # Reserve some time for future turns, but never exceed 80% of remaining
        time_budget = min(remaining * 0.9 / turns_left, remaining * 0.8)
        time_budget = max(time_budget, 0.01)  # at least 10ms

        # --- MCTS search ---
        best_move, mcts_value = self.mcts.search(board, time_budget)
        return best_move

    def commentate(self):
        if self.rat_belief is not None:
            best_pos = self.rat_belief.best_guess()
            best_prob = float(self.rat_belief.belief.max())
            return f"MCTS agent | Rat belief peak: {best_pos} ({best_prob:.1%})"
        return "MCTS agent"
