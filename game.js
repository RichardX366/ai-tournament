// game.js — JavaScript port of engine/game (Board, Worker, Move, Rat, History, enums)
// Uses BigInt for 64-bit bitboards.

// ===== Constants =====
const MAX_TURNS_PER_PLAYER = 40;
const BOARD_SIZE = 8;
const CARPET_POINTS_TABLE = { 1: -1, 2: 2, 3: 4, 4: 6, 5: 10, 6: 15, 7: 21 };
const ALLOWED_TIME = 240;
const RAT_BONUS = 4;
const RAT_PENALTY = 2;

// ===== Enums =====
const MoveType = Object.freeze({ PLAIN: 0, PRIME: 1, CARPET: 2, SEARCH: 3 });
const Cell = Object.freeze({ SPACE: 0, PRIMED: 1, CARPET: 2, BLOCKED: 3 });
const Noise = Object.freeze({ SQUEAK: 0, SCRATCH: 1, SQUEAL: 2 });
const Direction = Object.freeze({ UP: 0, RIGHT: 1, DOWN: 2, LEFT: 3 });
const Result = Object.freeze({ PLAYER: 0, ENEMY: 1, TIE: 2, ERROR: 3 });
const ResultArbiter = Object.freeze({
  PLAYER_A: 0,
  PLAYER_B: 1,
  TIE: 2,
  ERROR: 3,
});
const WinReason = Object.freeze({
  POINTS: 0,
  TIMEOUT: 1,
  INVALID_TURN: 2,
  CODE_CRASH: 3,
  MEMORY_ERROR: 4,
  FAILED_INIT: 5,
});

function locAfterDirection(loc, dir) {
  const [x, y] = loc;
  switch (dir) {
    case Direction.UP:
      return [x, y - 1];
    case Direction.DOWN:
      return [x, y + 1];
    case Direction.LEFT:
      return [x - 1, y];
    case Direction.RIGHT:
      return [x + 1, y];
    default:
      throw new Error(`Invalid direction: ${dir}`);
  }
}

// ===== Move =====
class Move {
  constructor(moveType, direction = null, roll = 0, searchLoc = null) {
    this.move_type = moveType;
    this.direction = direction;
    this.roll_length = roll;
    this.search_loc = searchLoc;
  }
  static plain(direction) {
    return new Move(MoveType.PLAIN, direction);
  }
  static prime(direction) {
    return new Move(MoveType.PRIME, direction);
  }
  static carpet(direction, roll) {
    return new Move(MoveType.CARPET, direction, roll);
  }
  static search(loc = null) {
    return new Move(MoveType.SEARCH, null, 0, loc);
  }

  toString() {
    const dn = ['UP', 'RIGHT', 'DOWN', 'LEFT'];
    switch (this.move_type) {
      case MoveType.PLAIN:
        return `PLAIN(${dn[this.direction]})`;
      case MoveType.PRIME:
        return `PRIME(${dn[this.direction]})`;
      case MoveType.CARPET:
        return `CARPET(${dn[this.direction]}, roll=${this.roll_length})`;
      case MoveType.SEARCH:
        return `SEARCH(loc=${this.search_loc})`;
      default:
        return 'UNKNOWN_MOVE';
    }
  }
}

// ===== Worker =====
class Worker {
  constructor(position, isPlayerA) {
    this.position = position;
    this.is_player_a = isPlayerA;
    this.is_player_b = !isPlayerA;
    this.points = 0;
    this.turns_left = MAX_TURNS_PER_PLAYER;
    this.time_left = ALLOWED_TIME;
  }
  get_location() {
    return this.position;
  }
  get_points() {
    return this.points;
  }
  increment_points(amount = 1) {
    this.points += amount;
  }
  decrement_points(amount = 1) {
    this.points -= amount;
  }
  copy() {
    const w = new Worker(this.position, this.is_player_a);
    w.points = this.points;
    w.turns_left = this.turns_left;
    w.time_left = this.time_left;
    return w;
  }
}

// ===== History =====
class History {
  constructor() {
    this.pos = [];
    this.rat_pos = [];
    this.left_behind_enums = [];
    this.a_points = [];
    this.b_points = [];
    this.a_turns_left = [];
    this.b_turns_left = [];
    this.a_time_left = [];
    this.b_time_left = [];
    this.rat_caught = [];
  }
  record_turn(board, move, ratCaught = false) {
    const playerAJustMoved = !board.is_player_a_turn;
    this.pos.push(board.player_worker.get_location());
    this.rat_caught.push(ratCaught);
    this.left_behind_enums.push(move.move_type);
    if (playerAJustMoved) {
      this.a_points.push(board.player_worker.get_points());
      this.b_points.push(board.opponent_worker.get_points());
      this.a_turns_left.push(board.player_worker.turns_left);
      this.b_turns_left.push(board.opponent_worker.turns_left);
      this.a_time_left.push(board.player_worker.time_left);
      this.b_time_left.push(board.opponent_worker.time_left);
    } else {
      this.a_points.push(board.opponent_worker.get_points());
      this.b_points.push(board.player_worker.get_points());
      this.a_turns_left.push(board.opponent_worker.turns_left);
      this.b_turns_left.push(board.player_worker.turns_left);
      this.a_time_left.push(board.opponent_worker.time_left);
      this.b_time_left.push(board.player_worker.time_left);
    }
  }
}

// ===== Board =====
const MASK64 = 0xffffffffffffffffn;
const CLEAR_TOP = 0x00ffffffffffffffn; // shift up  ((>>8))
const CLEAR_BOT = 0xffffffffffffff00n; // shift down ((<<8))
const CLEAR_RIGHT = 0x7f7f7f7f7f7f7f7fn; // shift left
const CLEAR_LEFT = 0xfefefefefefefefen; // shift right

class Board {
  constructor(timeToPlay = 20, buildHistory = false) {
    this.turn_count = 0;
    this.is_player_a_turn = true;
    this.winner = null;
    this.win_reason = null;
    this.time_to_play = timeToPlay;
    this.MAX_TURNS = MAX_TURNS_PER_PLAYER * 2;

    this._space_mask = MASK64;
    this._primed_mask = 0n;
    this._carpet_mask = 0n;
    this._blocked_mask = 0n;

    this.player_worker = new Worker([-1, -1], true);
    this.opponent_worker = new Worker([-1, -1], false);
    this.player_worker.time_left = timeToPlay;
    this.opponent_worker.time_left = timeToPlay;

    this.build_history = buildHistory;
    this.history = buildHistory ? new History() : null;

    this.opponent_search = [null, false];
    this.player_search = [null, false];

    this.valid_search_moves = [];
    for (let x = 0; x < BOARD_SIZE; x++) {
      for (let y = 0; y < BOARD_SIZE; y++) {
        this.valid_search_moves.push(Move.search([x, y]));
      }
    }
  }

  _loc_to_bit_index(loc) {
    return loc[1] * BOARD_SIZE + loc[0];
  }
  _bit(loc) {
    return 1n << BigInt(this._loc_to_bit_index(loc));
  }

  _shift_mask_up(mask) {
    return (mask >> 8n) & CLEAR_TOP;
  }
  _shift_mask_down(mask) {
    return (mask << 8n) & CLEAR_BOT;
  }
  _shift_mask_left(mask) {
    return (mask >> 1n) & CLEAR_RIGHT;
  }
  _shift_mask_right(mask) {
    return (mask << 1n) & CLEAR_LEFT;
  }

  is_valid_cell(loc) {
    return (
      loc[0] >= 0 && loc[1] >= 0 && loc[0] < BOARD_SIZE && loc[1] < BOARD_SIZE
    );
  }

  get_cell(loc) {
    if (!this.is_valid_cell(loc))
      throw new Error(`Invalid cell location: ${loc}`);
    const bm = this._bit(loc);
    if ((this._primed_mask & bm) !== 0n) return Cell.PRIMED;
    if ((this._carpet_mask & bm) !== 0n) return Cell.CARPET;
    if ((this._blocked_mask & bm) !== 0n) return Cell.BLOCKED;
    return Cell.SPACE;
  }

  set_cell(loc, cellType) {
    if (!this.is_valid_cell(loc))
      throw new Error(`Invalid cell location: ${loc}`);
    const bm = this._bit(loc);
    const inv = ~bm & MASK64;
    this._space_mask &= inv;
    this._primed_mask &= inv;
    this._carpet_mask &= inv;
    this._blocked_mask &= inv;
    switch (cellType) {
      case Cell.SPACE:
        this._space_mask |= bm;
        break;
      case Cell.PRIMED:
        this._primed_mask |= bm;
        break;
      case Cell.CARPET:
        this._carpet_mask |= bm;
        break;
      case Cell.BLOCKED:
        this._blocked_mask |= bm;
        break;
      default:
        throw new Error(`Invalid cell type: ${cellType}`);
    }
  }

  is_cell_blocked(loc) {
    if (!this.is_valid_cell(loc)) return true;
    const [ex, ey] = this.opponent_worker.get_location();
    if (ex === loc[0] && ey === loc[1]) return true;
    const [px, py] = this.player_worker.get_location();
    if (px === loc[0] && py === loc[1]) return true;
    const bm = this._bit(loc);
    return Boolean((this._blocked_mask | this._primed_mask) & bm);
  }

  is_cell_carpetable(loc) {
    if (!this.is_valid_cell(loc)) return false;
    const [ex, ey] = this.opponent_worker.get_location();
    if (ex === loc[0] && ey === loc[1]) return false;
    const [px, py] = this.player_worker.get_location();
    if (px === loc[0] && py === loc[1]) return false;
    return Boolean(this._primed_mask & this._bit(loc));
  }

  is_valid_move(move, enemy = false) {
    const worker = enemy ? this.opponent_worker : this.player_worker;
    const myLoc = worker.get_location();
    switch (move.move_type) {
      case MoveType.PLAIN: {
        const next = locAfterDirection(myLoc, move.direction);
        return !this.is_cell_blocked(next);
      }
      case MoveType.PRIME: {
        const next = locAfterDirection(myLoc, move.direction);
        if (this.is_cell_blocked(next)) return false;
        const bm = this._bit(myLoc);
        if (((this._primed_mask | this._carpet_mask) & bm) !== 0n) return false;
        return true;
      }
      case MoveType.CARPET: {
        if (move.roll_length < 1 || move.roll_length > BOARD_SIZE - 1)
          return false;
        let cur = myLoc;
        for (let i = 1; i <= move.roll_length; i++) {
          cur = locAfterDirection(cur, move.direction);
          if (!this.is_cell_carpetable(cur)) return false;
        }
        return true;
      }
      case MoveType.SEARCH:
        return this.is_valid_cell(move.search_loc);
    }
    return false;
  }

  get_valid_moves(enemy = false, excludeSearch = true) {
    const validMoves = [];
    const worker = enemy ? this.opponent_worker : this.player_worker;
    const myLoc = worker.get_location();
    const myBit = this._bit(myLoc);

    const playerBit = this._bit(this.player_worker.get_location());
    const enemyBit = this._bit(this.opponent_worker.get_location());
    const workersMask = playerBit | enemyBit;

    const blockedCells = this._blocked_mask | this._primed_mask | workersMask;
    const carpetableCells = this._primed_mask & ~workersMask;
    const canPrime = ((this._primed_mask | this._carpet_mask) & myBit) === 0n;

    const directions = [
      [Direction.UP, (m) => this._shift_mask_up(m)],
      [Direction.DOWN, (m) => this._shift_mask_down(m)],
      [Direction.LEFT, (m) => this._shift_mask_left(m)],
      [Direction.RIGHT, (m) => this._shift_mask_right(m)],
    ];

    for (const [dir, shift] of directions) {
      const nextCellMask = shift(myBit);
      if (nextCellMask !== 0n && (blockedCells & nextCellMask) === 0n) {
        validMoves.push(Move.plain(dir));
        if (canPrime) validMoves.push(Move.prime(dir));
      }
      let currentMask = myBit;
      for (let roll = 1; roll < BOARD_SIZE; roll++) {
        currentMask = shift(currentMask);
        if (currentMask === 0n) break;
        if ((carpetableCells & currentMask) === 0n) break;
        validMoves.push(Move.carpet(dir, roll));
      }
    }

    if (!excludeSearch) {
      for (const m of this.valid_search_moves) validMoves.push(m);
    }
    return validMoves;
  }

  forecast_move(move, checkOk = true) {
    const copy = this.get_copy();
    const ok = copy.apply_move(move, 0, checkOk);
    return ok ? copy : null;
  }

  apply_move(move, timer = 0, checkOk = true) {
    try {
      if (checkOk && !this.is_valid_move(move)) return false;

      switch (move.move_type) {
        case MoveType.PLAIN:
          this.player_worker.position = locAfterDirection(
            this.player_worker.get_location(),
            move.direction,
          );
          break;
        case MoveType.PRIME:
          this.set_cell(this.player_worker.get_location(), Cell.PRIMED);
          this.player_worker.position = locAfterDirection(
            this.player_worker.get_location(),
            move.direction,
          );
          this.player_worker.increment_points(1);
          break;
        case MoveType.CARPET: {
          let cur = this.player_worker.get_location();
          for (let i = 1; i <= move.roll_length; i++) {
            cur = locAfterDirection(cur, move.direction);
            this.set_cell(cur, Cell.CARPET);
          }
          this.player_worker.increment_points(
            CARPET_POINTS_TABLE[move.roll_length],
          );
          this.player_worker.position = cur;
          break;
        }
        case MoveType.SEARCH:
          // handled by game runner
          break;
      }
      this.end_turn(timer);
      return true;
    } catch (e) {
      return false;
    }
  }

  end_turn(timer = 0) {
    this.turn_count += 1;
    this.player_worker.turns_left -= 1;
    this.player_worker.time_left -= timer;
    this.check_win();
    this.is_player_a_turn = !this.is_player_a_turn;
  }

  check_win(timeoutBounds = 0.5) {
    if (this.player_worker.time_left <= 0) {
      if (this.opponent_worker.time_left <= timeoutBounds) {
        this.set_winner(Result.TIE, WinReason.TIMEOUT);
      } else {
        this.set_winner(Result.ENEMY, WinReason.TIMEOUT);
      }
    } else if (this.opponent_worker.time_left <= 0) {
      if (this.player_worker.time_left <= timeoutBounds) {
        this.set_winner(Result.TIE, WinReason.TIMEOUT);
      } else {
        this.set_winner(Result.PLAYER, WinReason.TIMEOUT);
      }
    } else if (
      (this.player_worker.turns_left === 0 &&
        this.opponent_worker.turns_left === 0) ||
      this.turn_count >= 2 * MAX_TURNS_PER_PLAYER
    ) {
      const p = this.player_worker.get_points();
      const o = this.opponent_worker.get_points();
      if (o > p) this.set_winner(Result.ENEMY, WinReason.POINTS);
      else if (o < p) this.set_winner(Result.PLAYER, WinReason.POINTS);
      else this.set_winner(Result.TIE, WinReason.POINTS);
    }
  }

  is_game_over() {
    return this.winner !== null;
  }

  get_copy(buildHistory = false) {
    const nb = new Board(this.time_to_play, buildHistory);
    nb.turn_count = this.turn_count;
    nb.is_player_a_turn = this.is_player_a_turn;
    nb.winner = this.winner;
    nb.win_reason = this.win_reason;
    nb._space_mask = this._space_mask;
    nb._primed_mask = this._primed_mask;
    nb._carpet_mask = this._carpet_mask;
    nb._blocked_mask = this._blocked_mask;
    nb.player_worker = this.player_worker.copy();
    nb.opponent_worker = this.opponent_worker.copy();
    nb.opponent_search = [...this.opponent_search];
    nb.player_search = [...this.player_search];
    return nb;
  }

  set_winner(result, reason) {
    this.winner = result;
    this.win_reason = reason;
  }
  get_winner() {
    return this.winner;
  }
  get_win_reason() {
    return this.win_reason;
  }
  get_history() {
    return this.history;
  }

  reverse_perspective() {
    const tmp = this.player_worker;
    this.player_worker = this.opponent_worker;
    this.opponent_worker = tmp;
  }
}

// ===== Rat =====
const HEADSTART_MOVES = 1000;

// [P(squeak), P(scratch), P(squeal)]
const NOISE_PROBS = {
  [Cell.BLOCKED]: [0.5, 0.3, 0.2],
  [Cell.SPACE]: [0.7, 0.15, 0.15],
  [Cell.PRIMED]: [0.1, 0.8, 0.1],
  [Cell.CARPET]: [0.1, 0.1, 0.8],
};
// [P(-1), P(correct), P(+1), P(+2)]
const DISTANCE_ERROR_PROBS = [0.12, 0.7, 0.12, 0.06];
const DISTANCE_ERROR_OFFSETS = [-1, 0, 1, 2];

function manhattanDistance(p1, p2) {
  return Math.abs(p1[0] - p2[0]) + Math.abs(p1[1] - p2[1]);
}

function cumulative(probs) {
  const out = [];
  let s = 0;
  for (const p of probs) {
    s += p;
    out.push(s);
  }
  return out;
}

class Rat {
  constructor(T) {
    // T[from_index][to_index] = transition probability
    const n = BOARD_SIZE * BOARD_SIZE;
    this.cumT = [];
    for (let i = 0; i < n; i++) {
      let s = 0;
      const row = [];
      for (const p of T[i]) {
        s += Number(p);
        row.push(s);
      }
      this.cumT.push(row);
    }
    this.noise_cum = {};
    for (const k of Object.keys(NOISE_PROBS)) {
      this.noise_cum[k] = cumulative(NOISE_PROBS[k]);
    }
    this.dist_cum = cumulative(DISTANCE_ERROR_PROBS);
    this.position = [0, 0];
  }

  _pos_to_index(pos) {
    return pos[1] * BOARD_SIZE + pos[0];
  }
  _index_to_pos(idx) {
    return [idx % BOARD_SIZE, Math.floor(idx / BOARD_SIZE)];
  }

  _sample3(cum) {
    const r = Math.random();
    if (r < cum[0]) return 0;
    if (r < cum[1]) return 1;
    return 2;
  }

  move() {
    const from = this._pos_to_index(this.position);
    const r = Math.random();
    const row = this.cumT[from];
    // binary search (bisect_left)
    let lo = 0,
      hi = row.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (row[mid] < r) lo = mid + 1;
      else hi = mid;
    }
    this.position = this._index_to_pos(lo);
  }

  make_noise(board) {
    const cellType = board.get_cell(this.position);
    const cum = this.noise_cum[cellType] || this.noise_cum[Cell.SPACE];
    return this._sample3(cum);
  }

  estimate_distance(workerPosition) {
    const actual = manhattanDistance(workerPosition, this.position);
    const r = Math.random();
    let offset = DISTANCE_ERROR_OFFSETS[DISTANCE_ERROR_OFFSETS.length - 1];
    for (let i = 0; i < this.dist_cum.length; i++) {
      if (r < this.dist_cum[i]) {
        offset = DISTANCE_ERROR_OFFSETS[i];
        break;
      }
    }
    const d = actual + offset;
    return d > 0 ? d : 0;
  }

  spawn() {
    this.position = [0, 0];
    for (let i = 0; i < HEADSTART_MOVES; i++) this.move();
  }

  get_position() {
    return this.position;
  }

  sample(board) {
    return [
      this.make_noise(board),
      this.estimate_distance(board.player_worker.get_location()),
    ];
  }
}

// ===== Exports =====
const Game = {
  // constants
  MAX_TURNS_PER_PLAYER,
  BOARD_SIZE,
  CARPET_POINTS_TABLE,
  ALLOWED_TIME,
  RAT_BONUS,
  RAT_PENALTY,
  // enums
  MoveType,
  Cell,
  Noise,
  Direction,
  Result,
  ResultArbiter,
  WinReason,
  // helpers
  locAfterDirection,
  manhattanDistance,
  // classes
  Move,
  Worker,
  History,
  Board,
  Rat,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = Game;
}
if (typeof window !== 'undefined') {
  // Expose under both names — viewer.html and minimax.js read window.GAME,
  // while older code may still use window.Game.
  window.Game = Game;
  window.GAME = Game;
}
