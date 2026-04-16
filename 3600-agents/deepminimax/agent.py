"""
agent.py — Expectiminimax agent with HMM rat tracking.

Strategy:
- Uses expectiminimax with alpha-beta pruning and iterative deepening
  to search the game tree for the best move.
- Maintains a probability distribution over rat positions using an HMM.
- Integrates rat search decisions directly into the tree search.

To use this agent, replace agent.py in your submission folder with this file.
The old MCTS agent is preserved as agent_mcts.py.
"""

from collections.abc import Callable
from typing import Tuple

from game.board import Board
from game.move import Move
from .expectiminimax import Expectiminimax
from .rat_belief import RatBelief


class PlayerAgent:
    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.rat_belief = None
        if transition_matrix is not None:
            self.rat_belief = RatBelief(transition_matrix)

        self.searcher = Expectiminimax(max_depth=10)
        self.turn_number = 0

    def play(
        self,
        board: Board,
        sensor_data: Tuple,
        time_left: Callable,
    ) -> Move:
        self.turn_number += 1
        noise, est_distance = sensor_data

        # ── Update rat belief ─────────────────────────────────────────────────
        if self.rat_belief is not None:
            self.rat_belief.update(board, int(noise), int(est_distance))

        # ── Time management (adaptive) ────────────────
        remaining = max(0.0, float(time_left()))
        turns_left = max(1, board.player_worker.turns_left)
        usable = max(0.15, remaining - 5.0)
        time_budget = min(5.5, max(0.15, 1.5 * usable / turns_left))

        # ── Search for best movement move ────────────────────────────────────
        best_move, best_val = self.searcher.search(
            board,
            self.rat_belief,
            time_budget,
        )

        # ── Rat search decision (skip-turn negamax comparison) ────────
        # Search = skip a turn positionally but gain search_ev in expectation.
        # Negamax the skipped board to get the full future cost of not moving.
        # If search_ev > skip_cost, the rat points outweigh the lost position.
        if self.rat_belief is not None:
            search_xy, search_ev = self.rat_belief.best_search_target()
            miss_ev = self.rat_belief.new_ev_if_miss()
            search_ev -= (
                miss_ev * (1 - self.rat_belief.belief.max()) * 0.3
            )  # Compare against the EV of not searching at all

            if search_ev > 0:
                skip_board = board.get_copy()
                skip_board.end_turn()
                skip_board.reverse_perspective()

                # Give the skip search a small time slice
                import time as _time

                self.searcher._deadline = _time.perf_counter() + time_budget * 0.5
                skip_val = -self.searcher._negamax(
                    skip_board,
                    max(1, self.searcher.max_depth - 1),
                    -float("inf"),
                    float("inf"),
                )

                skip_cost = best_val - skip_val
                if search_ev > skip_cost:
                    return Move.search(search_xy)

        return best_move

    def commentate(self):
        if self.rat_belief is not None:
            best_pos = self.rat_belief.best_guess()
            best_prob = float(self.rat_belief.belief.max())
            best_ev = self.rat_belief.ev_search(best_pos)
            nodes = self.searcher._nodes
            tt_hits = self.searcher._tt_hits
            tt_size = len(self.searcher._tt)
            return (
                f"Expectiminimax agent | "
                f"Rat: {best_pos} ({best_prob:.1%}, EV={best_ev:.2f}) | "
                f"Nodes: {nodes}, TT hits: {tt_hits}, TT size: {tt_size}"
            )
        return "Expectiminimax agent"
