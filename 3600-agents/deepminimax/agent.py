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

        # Depth 5 is a good starting point — iterative deepening means we
        # always finish depth 1-4 and only reach 5 if time allows.
        self.searcher = Expectiminimax(max_depth=11)
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

        # ── Search ───────────────────────────────────────────────────────────
        best_move, value = self.searcher.search(
            board,
            self.rat_belief,
            time_budget,
        )

        return best_move

    def commentate(self):
        if self.rat_belief is not None:
            best_pos = self.rat_belief.best_guess()
            best_prob = float(self.rat_belief.belief.max())
            best_ev = self.rat_belief.ev_search(best_pos)
            nodes = self.searcher._nodes
            return (
                f"Expectiminimax agent | "
                f"Rat: {best_pos} ({best_prob:.1%}, EV={best_ev:.2f}) | "
                f"Last search nodes: {nodes}"
            )
        return "Expectiminimax agent"
