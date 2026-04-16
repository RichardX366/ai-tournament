"""
rat_belief.py — JAX-accelerated HMM rat tracker.

Belief update models the exact game-loop sequence between our consecutive turns:

    1. Our previous search result (hit -> respawn reset, miss -> zero-out cell)
    2. Rat moves once (transition before opponent's sensor reading)
    3. Opponent's search result (hit -> respawn reset, miss -> zero-out cell)
    4. Rat moves once more (transition before our sensor reading)
    5. Observe: apply noise + distance likelihoods from sensor_data
    6. Normalise

The spawn prior is the stationary distribution of the transition matrix after
1000 steps from cell (0,0) — matching the engine's rat.spawn() exactly.

INDEXING CONVENTION (matches the engine)
-----------------------------------------
    flat_index = y * BOARD_SIZE + x
    (x, y)     = (index % BOARD_SIZE, index // BOARD_SIZE)
"""

import jax.numpy as jnp
from typing import Tuple

# ── constants ────────────────────────────────────────────────────────────────
BOARD_SIZE = 8
N = BOARD_SIZE * BOARD_SIZE  # 64

RAT_BONUS = 4
RAT_PENALTY = -2
HEADSTART_MOVES = 1000

# Emission table: P(noise | cell_type)
# Rows: SPACE=0, PRIMED=1, CARPET=2, BLOCKED=3
# Cols: SQUEAK=0, SCRATCH=1, SQUEAL=2
_EMISSION = jnp.array(
    [
        [0.70, 0.15, 0.15],  # SPACE
        [0.10, 0.80, 0.10],  # PRIMED
        [0.10, 0.10, 0.80],  # CARPET
        [0.50, 0.30, 0.20],  # BLOCKED
    ],
    dtype=jnp.float32,
)

# Distance error model
_DIST_OFFSETS = jnp.array([-1, 0, 1, 2], dtype=jnp.int32)
_DIST_PROBS = jnp.array([0.12, 0.70, 0.12, 0.06], dtype=jnp.float32)

# Pre-computed coordinate arrays
_ALL_IDX = jnp.arange(N, dtype=jnp.int32)
_ALL_X = _ALL_IDX % BOARD_SIZE
_ALL_Y = _ALL_IDX // BOARD_SIZE


def _xy_to_idx(x: int, y: int) -> int:
    return y * BOARD_SIZE + x


def _idx_to_xy(idx: int) -> Tuple[int, int]:
    return (int(idx % BOARD_SIZE), int(idx // BOARD_SIZE))


def _mask_to_bool_array(mask: int) -> jnp.ndarray:
    """Convert a Python int bitmask (up to 64 bits) to a (64,) bool JAX array."""
    return jnp.array([(mask >> i) & 1 for i in range(N)], dtype=jnp.bool_)


# ── main class ───────────────────────────────────────────────────────────────


class RatBelief:
    """
    HMM belief tracker for the rat's position.

    Models the full game-loop sequence: both players' search results,
    two rat transitions per round, and sensor observations.
    """

    def __init__(self, transition_matrix):
        # T^T stored so belief update is matmul: T_T @ belief
        T = jnp.array(transition_matrix, dtype=jnp.float32)
        self._T_T = T.T  # shape (64, 64)

        # Compute spawn prior: start at cell (0,0), run 1000 transitions.
        # This matches the engine's rat.spawn() exactly.
        spawn = jnp.zeros(N, dtype=jnp.float32).at[0].set(1.0)
        for _ in range(HEADSTART_MOVES):
            spawn = self._T_T @ spawn
        s = spawn.sum()
        self._spawn_prior = jnp.where(
            s > 0, spawn / s, jnp.ones(N, dtype=jnp.float32) / N
        )

        # Distribution one transition after a spawn (rat moves once before
        # the next sensor reading after a respawn).
        spawn_after_one = self._T_T @ self._spawn_prior
        s = spawn_after_one.sum()
        self._spawn_after_one = jnp.where(
            s > 0, spawn_after_one / s, self._spawn_prior
        )

        # Initial belief: rat spawned + 1000 moves, then moved once more
        # before we get our first sensor reading.
        self.belief: jnp.ndarray = self._spawn_after_one.copy()

        self._first_play = True
        self._entropy = 0.0

        # Cache for blocked mask
        self._cached_blocked_mask_val = None
        self._cached_blocked_arr = None

    # ── public API ───────────────────────────────────────────────────────────

    def update(self, board, noise, dist: int):
        """
        Call once at the top of every play() turn, before choosing a move.

        Follows the exact game-loop sequence:
          1. Handle our previous search result
          2. Rat transition (opponent's turn)
          3. Handle opponent's search result
          4. Rat transition (our turn — this is when sensor_data was taken)
          5. Apply sensor likelihoods (noise + distance)
          6. Normalise
        """
        my_loc, my_hit = board.player_search
        opp_loc, opp_hit = board.opponent_search

        if self._first_play:
            # First turn: belief is already spawn_after_one.
            # Skip steps 1-4, go straight to observation.
            self._first_play = False
        else:
            # ── 1. Our previous search result ────────────────────────────
            if my_loc is not None:
                if my_hit:
                    # We caught the rat — it respawned
                    self.belief = self._spawn_prior.copy()
                else:
                    # We missed — zero out that cell
                    idx = _xy_to_idx(*my_loc)
                    self.belief = self.belief.at[idx].set(0.0)
                    s = self.belief.sum()
                    self.belief = jnp.where(
                        s > 0, self.belief / s, self._spawn_prior
                    )

            # ── 2. Rat moves once (before opponent's sensor reading) ─────
            self.belief = self._T_T @ self.belief

            # ── 3. Opponent's search result ──────────────────────────────
            if opp_loc is not None:
                if opp_hit:
                    # Opponent caught the rat — it respawned
                    self.belief = self._spawn_prior.copy()
                else:
                    # Opponent missed — zero out that cell
                    idx = _xy_to_idx(*opp_loc)
                    self.belief = self.belief.at[idx].set(0.0)
                    s = self.belief.sum()
                    self.belief = jnp.where(
                        s > 0, self.belief / s, self._spawn_prior
                    )

            # ── 4. Rat moves once more (before our sensor reading) ───────
            self.belief = self._T_T @ self.belief

        # ── 5. Observation: noise + distance likelihood ──────────────────
        cell_types = self._get_cell_types(board)
        noise_lik = _EMISSION[cell_types, int(noise)]
        dist_lik = self._dist_likelihood(
            board.player_worker.get_location(), dist
        )
        likelihood = noise_lik * dist_lik

        # ── 6. Bayesian update + normalise ───────────────────────────────
        self.belief = self.belief * likelihood
        s = self.belief.sum()
        self.belief = jnp.where(s > 1e-12, self.belief / s, self._spawn_prior)

        # Compute entropy for adaptive search thresholds
        safe = jnp.maximum(self.belief, 1e-15)
        self._entropy = float(-(safe * jnp.log(safe)).sum())

    def reset(self):
        """Manually reset to spawn prior."""
        self.belief = self._spawn_prior.copy()
        self._first_play = False

    def best_guess(self) -> Tuple[int, int]:
        """(x, y) of the cell with highest belief probability."""
        return _idx_to_xy(int(jnp.argmax(self.belief)))

    def ev_search(self, xy: Tuple[int, int]) -> float:
        """EV of searching cell xy: 6p - 2."""
        p = float(self.belief[_xy_to_idx(*xy)])
        return 6.0 * p - 2.0

    def best_search_target(self) -> Tuple[Tuple[int, int], float]:
        """Return (xy, ev) for the highest-EV cell to search."""
        idx = int(jnp.argmax(self.belief))
        xy = _idx_to_xy(idx)
        return xy, self.ev_search(xy)

    def new_ev_if_miss(self) -> float:
        """EV of the best cell after the current best guess misses.
        Does not include the -2 penalty from the missed guess itself."""
        idx = int(jnp.argmax(self.belief))
        miss_belief = self.belief.at[idx].set(0.0)
        s = miss_belief.sum()
        if s > 1e-12:
            miss_belief = miss_belief / s
        else:
            miss_belief = self._spawn_prior
        new_best_p = float(miss_belief.max())
        return 6.0 * new_best_p - 2.0

    def should_search(self, threshold: float = 0.0) -> bool:
        """True if best search EV exceeds threshold."""
        _, best_ev = self.best_search_target()
        return best_ev > threshold

    @property
    def entropy(self) -> float:
        """Belief entropy — lower means more confident about rat position."""
        return self._entropy

    @property
    def best_prob(self) -> float:
        """Probability of the most likely rat cell."""
        return float(self.belief.max())

    # ── private helpers ──────────────────────────────────────────────────────

    def _get_cell_types(self, board) -> jnp.ndarray:
        """Extract cell types from board bitmasks into (64,) int32 array."""
        is_primed = _mask_to_bool_array(board._primed_mask)
        is_carpet = _mask_to_bool_array(board._carpet_mask)

        bm = board._blocked_mask
        if bm != self._cached_blocked_mask_val:
            self._cached_blocked_mask_val = bm
            self._cached_blocked_arr = _mask_to_bool_array(bm)
        is_blocked = self._cached_blocked_arr

        cell_types = jnp.zeros(N, dtype=jnp.int32)
        cell_types = jnp.where(is_blocked, jnp.int32(3), cell_types)
        cell_types = jnp.where(is_carpet, jnp.int32(2), cell_types)
        cell_types = jnp.where(is_primed, jnp.int32(1), cell_types)
        return cell_types

    def _dist_likelihood(
        self, worker_xy: Tuple[int, int], d_obs: int
    ) -> jnp.ndarray:
        """P(d_obs | rat at cell i, worker at worker_xy) for every cell."""
        wx, wy = worker_xy
        d_true = jnp.abs(_ALL_X - wx) + jnp.abs(_ALL_Y - wy)

        d_expected = jnp.maximum(d_true[None, :] + _DIST_OFFSETS[:, None], 0)
        matches = (d_expected == d_obs).astype(jnp.float32)
        likelihoods = (_DIST_PROBS[:, None] * matches).sum(axis=0)

        return jnp.maximum(likelihoods, 1e-9)
