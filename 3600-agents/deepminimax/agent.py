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
from game.enums import Cell, BOARD_SIZE, MoveType, Direction
from game.move import Move
from .expectiminimax import Expectiminimax
from .rat_belief import RatBelief


class PlayerAgent:
    def __init__(self, board, transition_matrix=None, time_left: Callable = None):  # type: ignore
        self.rat_belief = None
        if transition_matrix is not None:
            self.rat_belief = RatBelief(transition_matrix)

        self.searcher = Expectiminimax(max_depth=12)
        self.turn_number = 0
        self.rat = []

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
            self.rat += [self.rat_belief.belief]

        # ── Time management (adaptive) ────────────────
        remaining = max(0.0, float(time_left()))
        turns_left = max(1, board.player_worker.turns_left)
        usable = max(0.15, remaining - 5.0)
        time_budget = min(5.5, max(0.15, 1.5 * usable / turns_left))

        # It is the last turn and you are A
        if (
            board.player_worker.turns_left == 1
            and board.opponent_worker.turns_left == 1
        ):
            return self.last_move_a(board)

        # It is the last turn and you are B
        if (
            board.player_worker.turns_left == 1
            and board.opponent_worker.turns_left == 0
        ):
            return self.last_move_b(board)

        # ── Count surrounding tiles (manhattan distance 1–2) ─────────────────
        px, py = board.player_worker.get_location()
        surroundingTiles = 0
        surroundingCarpet = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) not in (1, 2):
                    continue
                nx, ny = px + dx, py + dy
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                    cell = board.get_cell((nx, ny))
                    if cell in (Cell.PRIMED, Cell.CARPET, Cell.SPACE):
                        surroundingTiles += 1
                    if cell == Cell.CARPET:
                        surroundingCarpet += 1

        if surroundingTiles > 0 and surroundingCarpet / surroundingTiles > 0.65:
            player_moves_1 = board.get_valid_moves()
            prime_moves = [m for m in player_moves_1 if m.move_type == MoveType.PRIME]
            carpet_moves = [
                m
                for m in player_moves_1
                if m.move_type == MoveType.CARPET and m.roll_length >= 2
            ]
            if carpet_moves:
                return self.search(board, time_budget)
            if prime_moves:
                player_moves_1 = prime_moves
            else:
                player_moves_1 = [
                    m for m in player_moves_1 if m.move_type == MoveType.PLAIN
                ]

            opponent_moves = self.get_next_n_enemy_moves(board, time_budget, n=2)

            move_sequences = []

            for move1 in player_moves_1:
                b = board.get_copy()
                m = b.apply_move(move1)
                if not m:
                    continue
                b.end_turn()
                b.reverse_perspective()
                m = b.apply_move(opponent_moves[0])
                if not m:
                    continue
                b.end_turn()
                b.reverse_perspective()

                player_moves_2 = b.get_valid_moves()
                prime_moves = [
                    m for m in player_moves_2 if m.move_type == MoveType.PRIME
                ]
                carpet_moves = [
                    m
                    for m in player_moves_2
                    if m.move_type == MoveType.CARPET and m.roll_length >= 2
                ]
                if carpet_moves:
                    return self.search(board, time_budget)
                if prime_moves:
                    player_moves_2 = prime_moves
                else:
                    player_moves_2 = [
                        m for m in player_moves_2 if m.move_type == MoveType.PLAIN
                    ]

                for move2 in player_moves_2:
                    b2 = b.get_copy()
                    m = b2.apply_move(move2)
                    if not m:
                        continue
                    b2.end_turn()
                    b2.reverse_perspective()
                    m = b2.apply_move(opponent_moves[1])
                    if not m:
                        continue
                    b2.end_turn()
                    b2.reverse_perspective()

                    best_move, best_val = self.searcher.search(
                        b2,
                        time_budget,
                    )

                    move_sequences.append((move1, move2, best_move, best_val))

            move_sequences.sort(key=lambda x: x[3], reverse=True)
            if move_sequences:
                return move_sequences[0][0]

        return self.search(board, time_budget)

    def last_move_a(self, board: Board) -> Move:
        """Return the best move for player A.

        Called when this agent is player A. Iterates all valid moves and picks
        the one that maximizes (player_points - opponent_points) after applying
        the move. Also considers searching for the rat if things are too bad.
        """
        moves = board.get_valid_moves()

        best_move = moves[0]
        best_val = -float("inf")

        for mv in moves:
            b = board.get_copy()
            b.apply_move(mv)
            b.end_turn()
            b.reverse_perspective()

            b_moves = b.get_valid_moves()
            passing_best_value = -float("inf")
            if not b_moves:
                continue
            for b_mv in b_moves:
                b2 = b.forecast_move(b_mv)
                if b2 is None:
                    continue
                val = b2.player_worker.get_points() - b2.opponent_worker.get_points()
                if val > passing_best_value:
                    passing_best_value = val
            passing_best_value = -passing_best_value
            if passing_best_value > best_val:
                best_val = passing_best_value
                best_move = mv

        if self.rat_belief is not None:
            b = board.get_copy()
            b.end_turn()
            b.reverse_perspective()
            moves = b.get_valid_moves()
            if not moves:
                return best_move
            passing_best_value = -float("inf")
            for mv in moves:
                b2 = b.forecast_move(mv)
                if b2 is None:
                    continue
                val = b2.player_worker.get_points() - b2.opponent_worker.get_points()
                if val > passing_best_value:
                    passing_best_value = val
            passing_best_value = -passing_best_value
            search_xy, search_ev = self.rat_belief.best_search_target()

            # If we win without guessing, just move
            if best_val > 0:
                return best_move

            # If we lose but can win by guessing, guess
            if best_val < 0 and passing_best_value > -4:
                return Move.search(search_xy)

            # If we tie by moving or passing, and the search EV is positive, search
            if best_val == 0 and passing_best_value == 0 and search_ev > 0:
                return Move.search(search_xy)

        return best_move

    def last_move_b(self, board: Board) -> Move:
        """Return the best move for player B.

        Called when this agent is player B. Iterates all valid moves and picks
        the one that maximizes (player_points - opponent_points) after applying
        the move. Also considers searching for the rat if things are too bad.
        """
        moves = board.get_valid_moves()

        best_move = moves[0]
        best_val = -float("inf")

        for mv in moves:
            b = board.forecast_move(mv)
            if b is None:
                continue
            val = b.player_worker.get_points() - b.opponent_worker.get_points()
            if val > best_val:
                best_val = val
                best_move = mv

        if self.rat_belief is not None:
            search_xy, search_ev = self.rat_belief.best_search_target()
            if best_val < 0:
                return Move.search(search_xy)
            elif best_val == 0 and search_ev > 0:
                return Move.search(search_xy)

        return best_move

    def get_next_n_enemy_moves(self, board: Board, time_budget: float, n: int):
        opponent_board = board.get_copy()
        opponent_board.end_turn()
        opponent_board.reverse_perspective()

        # Shallow search to predict opponent's next n moves.
        # This is because the player shouldn't impact the
        # opponent's move prediction too much when the
        # player is trapped/opponent is far.
        self.searcher.max_depth -= 2

        moves = []
        for _ in range(n):
            move = self.search(opponent_board, time_budget, search_rat=False)
            moves.append(move)
            opponent_board.apply_move(move)
            opponent_board.end_turn()
            opponent_board.reverse_perspective()
            opponent_board.end_turn()
            opponent_board.reverse_perspective()

        self.searcher.max_depth += 2

        return moves

    def search(self, board: Board, time_budget: float, search_rat: bool = True) -> Move:
        # ── Search for best movement move ────────────────────────────────────
        best_move, best_val = self.searcher.search(
            board,
            time_budget,
        )
        if self.searcher._timed_out:
            self.searcher.timeout_turns.append(self.turn_number)

        # ── Rat search decision (skip-turn negamax comparison) ────────
        # Search = skip a turn positionally but gain search_ev in expectation.
        # Negamax the skipped board to get the full future cost of not moving.
        # If search_ev > skip_cost, the rat points outweigh the lost position.
        if self.rat_belief is not None and search_rat:
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

            def fmt_val(v):
                v = round(float(v), 2)
                if v == 0.0:
                    return "0"
                s = f"{v:.2f}"
                if s.startswith("0."):
                    return s[1:]
                if s.startswith("-0."):
                    return "-" + s[2:]
                return s

            def fmt_list(x):
                if isinstance(x, list):
                    return "[" + ",".join(fmt_list(v) for v in x) + "]"
                return fmt_val(x)

            rat_str = "[" + ",".join(fmt_list(b.tolist()) for b in self.rat) + "]"
            timeout_str = str(self.searcher.timeout_turns) if self.searcher.timeout_turns else "[]"
            return rat_str + " timeouts:" + timeout_str
        timeout_str = str(self.searcher.timeout_turns) if self.searcher.timeout_turns else "[]"
        return "Expectiminimax agent timeouts:" + timeout_str
