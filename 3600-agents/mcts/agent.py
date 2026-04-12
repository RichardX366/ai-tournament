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
        self.rat_belief = None
        if transition_matrix is not None:
            self.rat_belief = RatBelief(transition_matrix)

        # Pass rat_belief into MCTS so the heuristic can use it
        self.mcts = MCTS(exploration=1.41, rollout_depth=10, rat_belief=self.rat_belief)
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
        if self.rat_belief is not None:
            self.rat_belief.update(board, int(noise), int(est_distance))

            # Evaluate whether to search
        search_move = None
        search_ev   = -float("inf")

        if self.rat_belief is not None:
            best_pos, best_ev = self.rat_belief.best_search_target()
            search_ev = best_ev

            turns_left = board.player_worker.turns_left

            # Search threshold scales with turns remaining:
            # - Early game: only search when very confident (EV > 1.0, p > 0.5)
            # - Late game: search more aggressively since there's less time to
            #   build carpet and every point matters (EV > 0, p > 1/3)
            if turns_left > 20:
                ev_threshold = 1.0
            elif turns_left > 10:
                ev_threshold = 0.5
            else:
                ev_threshold = 0.0   # any +EV search is worth it late game

            if best_ev > ev_threshold:
                search_move = Move.search(best_pos)

        # --- Decide: search vs. move ---
        # search_ev is in points (EV = 6p - 2).
        # If search EV > 2.0 points, prefer searching — skip MCTS entirely.
        if search_move is not None and search_ev > 2.0:
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

        # ── Compare search EV against MCTS value ─────────────────────────────
        # mcts_value is in [-1, 1]; convert search_ev (in points) to the same
        # scale for comparison. Rough conversion: 1 point ≈ 1/40 of the scale.
        if search_move is not None:
            search_value = search_ev / 40.0
            if search_value > mcts_value:
                return search_move

        return best_move

    def commentate(self):
        if self.rat_belief is not None:
            best_pos = self.rat_belief.best_guess()
            best_prob = float(self.rat_belief.belief.max())
            best_ev   = self.rat_belief.ev_search(best_pos)
            return (
                f"MCTS agent | Rat belief peak: {best_pos} "
                f"({best_prob:.1%}, EV={best_ev:.2f})"
            )
        return "MCTS agent"
    