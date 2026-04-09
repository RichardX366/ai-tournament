"""
rat_belief.py — JAX-accelerated HMM rat tracker

HOW TO USE
----------
In agent.py __init__:
    from .rat_belief import RatBelief
    self.rat_belief = RatBelief(transition_matrix)

In agent.py play(), call update() FIRST, then query:
    noise, dist = sensor_data
    self.rat_belief.update(board, noise, dist)

    belief              = self.rat_belief.belief               # jnp array shape (64,), sums to 1.0
    best_xy             = self.rat_belief.best_guess()         # (x, y) most likely rat cell
    best_cell, best_ev  = self.rat_belief.best_search_target() # best cell + its EV
    ev                  = self.rat_belief.ev_search((3, 4))    # EV of searching a specific cell
    go                  = self.rat_belief.should_search()      # True if best EV > 0

INDEXING CONVENTION (matches the engine)
-----------------------------------------
    flat_index = y * BOARD_SIZE + x
    (x, y)     = (index % BOARD_SIZE, index // BOARD_SIZE)
"""

import jax
import jax.numpy as jnp
from typing import Tuple

# ── constants (mirrored from enums.py / rat.py) ───────────────────────────────
BOARD_SIZE = 8
N          = BOARD_SIZE * BOARD_SIZE  # 64

RAT_BONUS   =  4   # points for a correct search
RAT_PENALTY = -2   # points for a wrong search

# ── module-level JAX constants (created once at import, never rebuilt) ────────

# Emission table: P(noise | cell_type)
# Rows → Cell type  : SPACE=0, PRIMED=1, CARPET=2, BLOCKED=3  (matches Cell enum)
# Cols → Noise type : SQUEAK=0, SCRATCH=1, SQUEAL=2           (matches Noise enum)
#                             SQK    SCR    SQL
_EMISSION = jnp.array([
    [0.70, 0.15, 0.15],   # SPACE
    [0.10, 0.80, 0.10],   # PRIMED
    [0.10, 0.10, 0.80],   # CARPET
    [0.50, 0.30, 0.20],   # BLOCKED
], dtype=jnp.float32)     # shape (4, 3)

# Distance error model: d_obs = clamp(d_true + offset, 0)
# P(offset = -1) = 0.12,  P(0) = 0.70,  P(+1) = 0.12,  P(+2) = 0.06
_DIST_OFFSETS = jnp.array([-1, 0, 1, 2], dtype=jnp.int32)              # shape (4,)
_DIST_PROBS   = jnp.array([0.12, 0.70, 0.12, 0.06], dtype=jnp.float32) # shape (4,)

# (x, y) coordinates for every cell index — precomputed once
_ALL_IDX = jnp.arange(N, dtype=jnp.int32)
_ALL_X   = _ALL_IDX % BOARD_SIZE   # shape (64,)
_ALL_Y   = _ALL_IDX // BOARD_SIZE  # shape (64,)


# ── helpers ───────────────────────────────────────────────────────────────────

def _xy_to_idx(x: int, y: int) -> int:
    return y * BOARD_SIZE + x

def _idx_to_xy(idx: int) -> Tuple[int, int]:
    return (int(idx % BOARD_SIZE), int(idx // BOARD_SIZE))

def _mask_to_bool_array(mask: int) -> jnp.ndarray:
    """
    Convert a Python int bitmask (up to 64 bits) to a (64,) bool JAX array.
    Bit i is set iff cell i belongs to this mask.
    Done in pure Python to avoid int64 overflow issues in JAX on 32-bit platforms.
    """
    return jnp.array([(mask >> i) & 1 for i in range(N)], dtype=jnp.bool_)


# ── main class ────────────────────────────────────────────────────────────────

class RatBelief:
    """
    HMM belief tracker for the rat's position.

    Each turn the HMM does:
        1. Predict   : belief = T^T @ belief   (rat moves before it makes noise)
        2. Likelihood: P(noise|cell) * P(dist_obs|cell, worker_pos)  for every cell
        3. Update    : belief *= likelihood,  then renormalise

    Also handles:
        - Auto-reset to uniform prior when the rat is caught (either player)
        - Zeroing out cells where a search confirmed the rat is NOT present
    """

    def __init__(self, transition_matrix):
        """
        Parameters
        ----------
        transition_matrix : array-like, shape (64, 64)
            T[i][j] = P(rat moves from cell i to cell j).
            Rows sum to 1.0.  Index convention: i = y*8 + x.
        """
        # T^T stored so belief update is just a matmul: T_T @ belief
        # belief_new[j] = sum_i T[i,j] * belief[i]  =  (T^T @ belief)[j]
        self._T_T = jnp.array(transition_matrix, dtype=jnp.float32).T  # shape (64, 64)

        # Uniform prior — after 1000 headstart steps we have no information
        self.belief: jnp.ndarray = jnp.ones(N, dtype=jnp.float32) / N
        self._spawn_prior        = self.belief  # reused on every respawn

        # Flag set when a rat catch is detected; triggers reset on the next turn
        self._rat_was_caught = False

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, board, noise, dist: int):
        """
        Call once at the top of every play() turn, before choosing a move.

        Parameters
        ----------
        board : Board   — engine Board object
        noise : Noise   — Noise enum (or raw int: SQUEAK=0, SCRATCH=1, SQUEAL=2)
        dist  : int     — noisy manhattan distance estimate from sensor_data
        """
        # ── 0. Respawn reset (deferred one turn so the catch observation lands) ──
        if self._rat_was_caught:
            self.belief          = self._spawn_prior
            self._rat_was_caught = False

        opp_loc, opp_hit = board.opponent_search
        _,       my_hit  = board.player_search

        if opp_hit or my_hit:
            # Rat was just caught — new rat spawned, reset to uniform prior
            self._rat_was_caught = True
            self.belief          = self._spawn_prior
            return

        # ── 1. Zero out confirmed-miss cell from opponent's last search ──────
        if opp_loc is not None and not opp_hit:
            idx         = _xy_to_idx(*opp_loc)
            self.belief = self.belief.at[idx].set(0.0)
            s           = self.belief.sum()
            self.belief = jnp.where(s > 0, self.belief / s, self._spawn_prior)

        # ── 2. Predict: rat moves once before making noise ───────────────────
        self.belief = self._T_T @ self.belief

        # ── 3. Likelihood from noise and distance observations ───────────────
        cell_types = self._get_cell_types(board)                             # (64,) int32
        noise_lik  = self._noise_likelihood(cell_types, int(noise))          # (64,)
        dist_lik   = self._dist_likelihood(board.player_worker.get_location(), dist)  # (64,)
        likelihood = noise_lik * dist_lik                                    # (64,)

        # ── 4. Update and renormalise ────────────────────────────────────────
        self.belief = self.belief * likelihood
        s           = self.belief.sum()
        self.belief = jnp.where(s > 1e-12, self.belief / s, self._spawn_prior)

    def reset(self):
        """Manually reset to uniform prior (e.g. if you catch the rat yourself)."""
        self.belief          = self._spawn_prior
        self._rat_was_caught = False

    def best_guess(self) -> Tuple[int, int]:
        """(x, y) of the cell with the highest belief probability."""
        return _idx_to_xy(int(jnp.argmax(self.belief)))

    def ev_search(self, xy: Tuple[int, int]) -> float:
        """
        Expected value of a Search move at cell xy.

            EV = p * RAT_BONUS + (1-p) * RAT_PENALTY
               = p * 4 + (1-p) * (-2)
               = 6p - 2

        EV > 0 when p > 1/3.
        """
        p = float(self.belief[_xy_to_idx(*xy)])
        return 6.0 * p - 2.0

    def best_search_target(self) -> Tuple[Tuple[int, int], float]:
        """
        Return (xy, ev) for the highest-EV cell to search.
        Since EV = 6p-2 is monotone in p, this is just argmax of belief.
        """
        idx = int(jnp.argmax(self.belief))
        xy  = _idx_to_xy(idx)
        return xy, self.ev_search(xy)

    def should_search(self, threshold: float = 0.0) -> bool:
        """True if the best search cell has EV > threshold (default: any +EV)."""
        _, best_ev = self.best_search_target()
        return best_ev > threshold

    # ── private helpers ───────────────────────────────────────────────────────

    def _get_cell_types(self, board) -> jnp.ndarray:
        """
        Extract cell types from the board's bitmasks into a (64,) int32 JAX array.

        Cell enum values: SPACE=0, PRIMED=1, CARPET=2, BLOCKED=3.

        The board stores state as three 64-bit Python ints. We extract each bit
        in Python (avoiding JAX int64 platform issues) then hand a bool array
        to JAX for the vectorised where() assignments.
        """
        is_primed  = _mask_to_bool_array(board._primed_mask)   # (64,) bool
        is_carpet  = _mask_to_bool_array(board._carpet_mask)   # (64,) bool
        is_blocked = _mask_to_bool_array(board._blocked_mask)  # (64,) bool

        # Priority order matches board.get_cell(): PRIMED > CARPET > BLOCKED > SPACE
        cell_types = jnp.zeros(N, dtype=jnp.int32)                       # SPACE=0
        cell_types = jnp.where(is_blocked, jnp.int32(3), cell_types)     # BLOCKED=3
        cell_types = jnp.where(is_carpet,  jnp.int32(2), cell_types)     # CARPET=2
        cell_types = jnp.where(is_primed,  jnp.int32(1), cell_types)     # PRIMED=1

        return cell_types  # shape (64,)

    def _noise_likelihood(self, cell_types: jnp.ndarray, noise_idx: int) -> jnp.ndarray:
        """
        P(noise | rat at cell i) for every cell i.
        Looks up the emission table by cell type and noise index.
        """
        return _EMISSION[cell_types, noise_idx]  # shape (64,)

    def _dist_likelihood(self, worker_xy: Tuple[int, int], d_obs: int) -> jnp.ndarray:
        """
        P(d_obs | rat at cell i, worker at worker_xy) for every cell i.

        For each cell:
            d_true    = manhattan_distance(worker, cell)
            d_clamped = max(d_true + offset, 0)   for each of the 4 offsets
            likelihood += P(offset) * (d_clamped == d_obs)

        The clamp mirrors the engine: estimate_distance() in rat.py returns max(d+offset, 0).
        Broadcast shape: offsets (4,1) over cells (1,64) → (4,64), then sum over offsets.
        """
        wx, wy = worker_xy
        d_true = jnp.abs(_ALL_X - wx) + jnp.abs(_ALL_Y - wy)          # (64,)

        # (4, 64): clamped expected observation for each offset × each cell
        d_expected  = jnp.maximum(d_true[None, :] + _DIST_OFFSETS[:, None], 0)
        matches     = (d_expected == d_obs).astype(jnp.float32)         # (4, 64)
        likelihoods = (_DIST_PROBS[:, None] * matches).sum(axis=0)      # (64,)

        # Epsilon floor keeps belief from collapsing to all-zeros on model mismatch
        return jnp.maximum(likelihoods, 1e-9)                           # (64,)
    