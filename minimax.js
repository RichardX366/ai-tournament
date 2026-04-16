// minimax.js — JS port of 3600-agents/minimax expectiminimax search.
// Classic script — expects game.js to have been loaded first (earlier <script> tag).
// Exposes window.MINIMAX with:
//   evaluate(board)                          — static eval (no rat belief)
//   Expectiminimax                           — negamax search class
//   rankMoves(board, { maxDepth, timeBudget }) — ranked list of root moves
//                                               [{ move, score, label }]
//   moveLabel(move)                          — human-readable move string

(function () {
  const G =
    (typeof window !== 'undefined' && (window.GAME || window.Game)) ||
    (typeof module !== 'undefined' && typeof require !== 'undefined'
      ? require('./game.js')
      : null);
  if (!G) {
    throw new Error(
      'minimax.js: game.js must be loaded first (window.GAME is undefined).',
    );
  }

  const {
    MoveType,
    Direction,
    Result,
    BOARD_SIZE,
    CARPET_POINTS_TABLE,
    Move,
    locAfterDirection,
  } = G;

  const INF = Infinity;

  // --- Carpet evaluation table (matches expectiminimax.py) ---------------
  const CARPET_PTS = [0, -1, 2, 4, 6, 10, 15, 21];
  const CARPET_EVALUATIONS = [0.0, -1.0];
  for (let L = 2; L < 8; L++)
    CARPET_EVALUATIONS.push((5 * CARPET_PTS[L] - 1) / 6);

  const DIRECTIONS = [
    Direction.UP,
    Direction.RIGHT,
    Direction.DOWN,
    Direction.LEFT,
  ];
  const DIR_MOVEMENTS = {
    [Direction.UP]: [0, -1],
    [Direction.DOWN]: [0, 1],
    [Direction.LEFT]: [-1, 0],
    [Direction.RIGHT]: [1, 0],
  };
  const OPPOSITE = {
    [Direction.UP]: Direction.DOWN,
    [Direction.DOWN]: Direction.UP,
    [Direction.LEFT]: Direction.RIGHT,
    [Direction.RIGHT]: Direction.LEFT,
  };

  // Neighbor offsets (for openness calc)
  const NEIGHBOR_OFFSETS = [
    [0, -1],
    [1, 0],
    [0, 1],
    [-1, 0],
  ];
  const NEIGHBOR2 = NEIGHBOR_OFFSETS.map(([dx, dy]) => ({
    dx,
    dy,
    nbr2: NEIGHBOR_OFFSETS.map(([dx2, dy2]) => [dx + dx2, dy + dy2]),
  }));

  function bitAt(x, y) {
    return 1n << BigInt(y * 8 + x);
  }

  // --- Line freedom: primable corridor potential -------------------------
  function lineFreedom(board, loc, playerLoc, oppLoc) {
    const [x, y] = loc;
    const blocked = board._blocked_mask;
    const primed = board._primed_mask;
    const carpet = board._carpet_mask;
    const occupied = blocked | primed | carpet;
    const [px, py] = playerLoc;
    const [ox, oy] = oppLoc;

    let total = 0;
    let bestRunway = 0;

    const axes = [
      [[0, -1], [0, 1]],   // vertical: UP / DOWN
      [[-1, 0], [1, 0]],   // horizontal: LEFT / RIGHT
    ];

    for (const [[dx1, dy1], [dx2, dy2]] of axes) {
      let count1 = 0;
      let cx = x + dx1, cy = y + dy1;
      while (cx >= 0 && cx < 8 && cy >= 0 && cy < 8) {
        const bit = bitAt(cx, cy);
        if ((occupied & bit) || (cx === px && cy === py) || (cx === ox && cy === oy)) break;
        count1++;
        cx += dx1; cy += dy1;
      }

      let count2 = 0;
      cx = x + dx2; cy = y + dy2;
      while (cx >= 0 && cx < 8 && cy >= 0 && cy < 8) {
        const bit = bitAt(cx, cy);
        if ((occupied & bit) || (cx === px && cy === py) || (cx === ox && cy === oy)) break;
        count2++;
        cx += dx2; cy += dy2;
      }

      const selfOk = (occupied & bitAt(x, y)) ? 0 : 1;
      const runway = count1 + count2 + selfOk;

      if (runway > bestRunway) bestRunway = runway;
      if (runway >= 2) total += CARPET_EVALUATIONS[Math.min(runway, 7)];
    }

    return { total, bestRunway };
  }

  // --- Line extension value (port of _ext) -------------------------------
  function ext(board, loc, playerLoc, oppLoc) {
    const [x, y] = loc;
    const bit = bitAt(x, y);
    const primed = board._primed_mask;
    const carpet = board._carpet_mask;
    const blocked = board._blocked_mask;
    if ((primed | carpet | blocked) & bit) return 0;

    let best = 0;
    for (const dir of DIRECTIONS) {
      const [dx, dy] = DIR_MOVEMENTS[dir];
      const nx = x + dx,
        ny = y + dy;
      if (!(nx >= 0 && nx < 8 && ny >= 0 && ny < 8)) continue;
      const nextBit = bitAt(nx, ny);
      if ((blocked | primed) & nextBit) continue;
      if (
        (nx === playerLoc[0] && ny === playerLoc[1]) ||
        (nx === oppLoc[0] && ny === oppLoc[1])
      )
        continue;

      const oppDir = OPPOSITE[dir];
      const [odx, ody] = DIR_MOVEMENTS[oppDir];
      let primedBehind = 0;
      let bx = x + odx,
        by = y + ody;
      while (bx >= 0 && bx < 8 && by >= 0 && by < 8 && primed & bitAt(bx, by)) {
        primedBehind++;
        bx += odx;
        by += ody;
      }
      let availableAhead = 0;
      let ax = nx + dx,
        ay = ny + dy;
      const space = board._space_mask;
      while (ax >= 0 && ax < 8 && ay >= 0 && ay < 8) {
        const ab = bitAt(ax, ay);
        if (space & ab && !((primed | carpet | blocked) & ab)) {
          availableAhead++;
          ax += dx;
          ay += dy;
        } else break;
      }
      let val;
      if (primedBehind > 0) {
        val =
          CARPET_EVALUATIONS[Math.min(primedBehind + 1, 7)] * 0.5 +
          availableAhead * 0.15;
      } else {
        val = availableAhead * 0.2;
      }
      if (val > best) best = val;
    }
    return best;
  }

  // --- Static evaluation -------------------------------------------------
  function evaluate(board) {
    if (board.is_game_over()) {
      const w = board.get_winner();
      if (w === Result.PLAYER) return 999;
      if (w === Result.ENEMY) return -999;
      return 0;
    }
    const player = board.player_worker;
    const opponent = board.opponent_worker;
    const [px, py] = player.get_location();
    const [ox, oy] = opponent.get_location();

    const scoreDiff = player.get_points() - opponent.get_points();
    const turnDiff = player.turns_left - opponent.turns_left;

    const playerMoves = board.get_valid_moves(false, true);
    const opponentMoves = board.get_valid_moves(true, true);

    function tact(moves) {
      let bestCarpet = 0,
        carpetSum = 0,
        primeCount = 0;
      for (const mv of moves) {
        if (mv.move_type === MoveType.CARPET) {
          const rl = mv.roll_length;
          const pts =
            rl >= 0 && rl < CARPET_EVALUATIONS.length
              ? CARPET_EVALUATIONS[rl]
              : 0;
          if (pts > bestCarpet) bestCarpet = pts;
          if (pts > 0) carpetSum += pts;
        } else if (mv.move_type === MoveType.PRIME) {
          primeCount++;
        }
      }
      return [bestCarpet, carpetSum, primeCount];
    }
    const [pBest, pSum, pPrime] = tact(playerMoves);
    const [oBest, oSum, oPrime] = tact(opponentMoves);

    // Openness
    const blocked = board._blocked_mask;
    let playerOpen = 0,
      oppOpen = 0;
    for (const { dx, dy, nbr2 } of NEIGHBOR2) {
      let nx = px + dx,
        ny = py + dy;
      if (
        nx >= 0 &&
        nx < 8 &&
        ny >= 0 &&
        ny < 8 &&
        !(blocked & bitAt(nx, ny))
      ) {
        playerOpen += 2;
        for (const [dx2, dy2] of nbr2) {
          const nx2 = px + dx2,
            ny2 = py + dy2;
          if (
            nx2 >= 0 &&
            nx2 < 8 &&
            ny2 >= 0 &&
            ny2 < 8 &&
            !(blocked & bitAt(nx2, ny2))
          ) {
            playerOpen += 1;
          }
        }
      }
      nx = ox + dx;
      ny = oy + dy;
      if (
        nx >= 0 &&
        nx < 8 &&
        ny >= 0 &&
        ny < 8 &&
        !(blocked & bitAt(nx, ny))
      ) {
        oppOpen += 2;
        for (const [dx2, dy2] of nbr2) {
          const nx2 = ox + dx2,
            ny2 = oy + dy2;
          if (
            nx2 >= 0 &&
            nx2 < 8 &&
            ny2 >= 0 &&
            ny2 < 8 &&
            !(blocked & bitAt(nx2, ny2))
          ) {
            oppOpen += 1;
          }
        }
      }
    }

    const playerCenter = 7 - (Math.abs(px - 3.5) + Math.abs(py - 3.5));
    const oppCenter = 7 - (Math.abs(ox - 3.5) + Math.abs(oy - 3.5));
    const playerExt = ext(board, [px, py], [px, py], [ox, oy]);
    const oppExt = ext(board, [ox, oy], [px, py], [ox, oy]);

    // Line freedom: mid-game only (ramp in 10→20, full 20→55, ramp out 55→65)
    const tc = board.turn_count;
    let lfWeight;
    if (tc < 10)       lfWeight = 0;
    else if (tc < 20)  lfWeight = (tc - 10) / 10;
    else if (tc < 55)  lfWeight = 1;
    else if (tc < 65)  lfWeight = (65 - tc) / 10;
    else               lfWeight = 0;

    let pLFtotal = 0, oLFtotal = 0, pLFbest = 0, oLFbest = 0;
    if (lfWeight > 0) {
      const pLF = lineFreedom(board, [px, py], [px, py], [ox, oy]);
      const oLF = lineFreedom(board, [ox, oy], [px, py], [ox, oy]);
      pLFtotal = pLF.total;  oLFtotal = oLF.total;
      pLFbest = pLF.bestRunway; oLFbest = oLF.bestRunway;
    }

    // Denial: when ahead in late game, penalize opponent's positional terms more
    const turnsLeft = Math.max(1, player.turns_left);
    let denial = 0;
    if (scoreDiff > 0 && turnsLeft <= 20) {
      denial = Math.min(0.5, 0.025 * (20 - turnsLeft) * Math.min(scoreDiff, 12) / 12);
    }

    return (
      3.0 * scoreDiff +
      0.12 * turnDiff +
      0.28 * (playerMoves.length - opponentMoves.length) +
      0.95 * pBest - (0.95 + denial) * oBest +
      0.18 * (pSum - oSum) +
      0.35 * (pPrime - oPrime) +
      0.22 * (playerOpen - oppOpen) +
      0.14 * (playerCenter - oppCenter) +
      0.45 * playerExt - (0.45 + denial) * oppExt +
      0.30 * lfWeight * pLFtotal - (0.30 + denial * 0.5) * lfWeight * oLFtotal +
      0.20 * lfWeight * pLFbest - (0.20 + denial * 0.3) * lfWeight * oLFbest
    );
  }

  // --- Move ordering -----------------------------------------------------
  function orderMovesFast(board, moves) {
    const [px, py] = board.player_worker.get_location();
    const primed = board._primed_mask;
    const scored = moves.map((mv) => {
      let score = 0;
      if (mv.move_type === MoveType.CARPET) {
        const rl = mv.roll_length;
        const pts =
          rl >= 0 && rl < CARPET_EVALUATIONS.length
            ? CARPET_EVALUATIONS[rl]
            : 0;
        score = pts * 3;
      } else if (mv.move_type === MoveType.PRIME) {
        score = 1;
        const opp = OPPOSITE[mv.direction];
        const [odx, ody] = DIR_MOVEMENTS[opp];
        const bx = px + odx,
          by = py + ody;
        if (bx >= 0 && bx < 8 && by >= 0 && by < 8 && primed & bitAt(bx, by)) {
          score += 4;
        }
      }
      return { score, mv };
    });
    scored.sort((a, b) => b.score - a.score);
    return scored.map((s) => s.mv);
  }

  function orderMovesFull(board, moves, evalFn) {
    const scored = [];
    for (const mv of moves) {
      const child = board.forecast_move(mv, true);
      if (!child) continue;
      child.reverse_perspective();
      let pts = 0;
      if (mv.move_type === MoveType.PRIME) pts = 0.5;
      else if (mv.move_type === MoveType.CARPET) {
        const rl = mv.roll_length;
        pts =
          rl >= 0 && rl < CARPET_EVALUATIONS.length
            ? CARPET_EVALUATIONS[rl]
            : 0;
      }
      const score = -evalFn(child) + 0.55 * pts;
      scored.push({ score, mv });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.map((s) => s.mv);
  }

  // --- Expectiminimax search --------------------------------------------
  class Expectiminimax {
    constructor(maxDepth = 4) {
      this.maxDepth = maxDepth;
      this._nodes = 0;
      this._deadline = Infinity;
    }

    _now() {
      return typeof performance !== 'undefined'
        ? performance.now()
        : Date.now();
    }

    search(board, timeBudgetMs = 500) {
      this._deadline = this._now() + timeBudgetMs;
      this._nodes = 0;

      const moves = board.get_valid_moves(false, true);
      if (!moves.length) return { move: Move.plain(Direction.UP), value: 0 };

      let ordered = orderMovesFull(board, moves, (b) => evaluate(b));
      if (!ordered.length) return { move: moves[0], value: 0 };

      let bestMove = ordered[0];
      let bestValue = -INF;

      for (let depth = 1; depth <= this.maxDepth; depth++) {
        if (this._now() >= this._deadline) break;
        let alpha = -INF;
        const beta = INF;
        let dBest = ordered[0];
        let dVal = -INF;
        let done = true;

        for (let i = 0; i < ordered.length; i++) {
          if (this._now() >= this._deadline) {
            done = false;
            break;
          }
          const mv = ordered[i];
          const child = board.forecast_move(mv, true);
          if (!child) continue;
          child.reverse_perspective();

          let val;
          if (i === 0) {
            val = -this._negamax(child, depth - 1, -beta, -alpha);
          } else {
            val = -this._negamax(child, depth - 1, -alpha - 0.01, -alpha);
            if (val > alpha && val < beta) {
              val = -this._negamax(child, depth - 1, -beta, -alpha);
            }
          }
          if (val > dVal) {
            dVal = val;
            dBest = mv;
          }
          if (val > alpha) alpha = val;
        }

        if (dVal > -1e17) {
          bestValue = dVal;
          bestMove = dBest;
          const i = ordered.indexOf(dBest);
          if (i > 0) {
            ordered.splice(i, 1);
            ordered.unshift(dBest);
          }
        }
        if (!done) break;
      }

      return { move: bestMove, value: bestValue };
    }

    _negamax(board, depth, alpha, beta) {
      this._nodes++;
      if (depth <= 0 || board.is_game_over() || this._now() >= this._deadline) {
        return evaluate(board);
      }
      const moves = board.get_valid_moves(false, true);
      if (!moves.length) return evaluate(board);
      const ordered = orderMovesFast(board, moves);

      let best = -INF;
      for (let i = 0; i < ordered.length; i++) {
        if (this._now() >= this._deadline) break;
        const mv = ordered[i];
        const child = board.forecast_move(mv, true);
        if (!child) continue;
        child.reverse_perspective();

        let val;
        if (i === 0) {
          val = -this._negamax(child, depth - 1, -beta, -alpha);
        } else {
          val = -this._negamax(child, depth - 1, -alpha - 0.01, -alpha);
          if (val > alpha && val < beta) {
            val = -this._negamax(child, depth - 1, -beta, -alpha);
          }
        }
        if (val > best) best = val;
        if (val > alpha) alpha = val;
        if (alpha >= beta) break;
      }
      return best > -1e17 ? best : evaluate(board);
    }
  }

  // --- PV-returning negamax -----------------------------------------------
  // Returns { value, pv: [ { label, side } , ... ] }
  // Each pv entry records the move label and which side played it.
  // "side" alternates: 0 = side-to-move at that node, 1 = opponent, …
  function negamaxPV(board, depth, alpha, beta, deadline, ply) {
    if (
      depth <= 0 ||
      board.is_game_over() ||
      (typeof performance !== 'undefined' ? performance.now() : Date.now()) >=
        deadline
    ) {
      return { value: evaluate(board), pv: [] };
    }
    const moves = board.get_valid_moves(false, true);
    if (!moves.length) return { value: evaluate(board), pv: [] };
    const ordered = orderMovesFast(board, moves);

    let best = -INF;
    let bestPV = [];
    for (let i = 0; i < ordered.length; i++) {
      if (
        (typeof performance !== 'undefined'
          ? performance.now()
          : Date.now()) >= deadline
      )
        break;
      const mv = ordered[i];
      const child = board.forecast_move(mv, true);
      if (!child) continue;
      child.reverse_perspective();

      const res = negamaxPV(child, depth - 1, -beta, -alpha, deadline, ply + 1);
      const val = -res.value;

      if (val > best) {
        best = val;
        bestPV = [{ label: moveLabel(mv), side: ply % 2 }, ...res.pv];
      }
      if (val > alpha) alpha = val;
      if (alpha >= beta) break;
    }
    if (best <= -1e17) return { value: evaluate(board), pv: [] };
    return { value: best, pv: bestPV };
  }

  // --- Ranked root moves for UI -----------------------------------------
  // Returns [{ move, score, label, pv }] sorted by score.
  // pv is the principal variation: array of { label, side } where
  // side 0 = current player, side 1 = opponent.
  function rankMoves(board, opts = {}) {
    const maxDepth = opts.maxDepth ?? 3;
    const timeBudget = opts.timeBudget ?? 400; // ms
    const moves = board.get_valid_moves(false, true);
    if (!moves.length) return [];

    const now =
      typeof performance !== 'undefined'
        ? performance.now()
        : Date.now();
    const deadline = now + timeBudget;

    const results = [];
    for (const mv of moves) {
      const child = board.forecast_move(mv, true);
      if (!child) continue;
      child.reverse_perspective();
      const res = negamaxPV(child, maxDepth - 1, -INF, INF, deadline, 1);
      const val = -res.value;
      results.push({
        move: mv,
        score: val,
        label: moveLabel(mv),
        pv: [{ label: moveLabel(mv), side: 0 }, ...res.pv],
      });
    }
    results.sort((a, b) => b.score - a.score);
    return results;
  }

  function moveLabel(move) {
    const dn = ['UP', 'RIGHT', 'DOWN', 'LEFT'];
    switch (move.move_type) {
      case MoveType.PLAIN:
        return `PLAIN ${dn[move.direction]}`;
      case MoveType.PRIME:
        return `PRIME ${dn[move.direction]}`;
      case MoveType.CARPET:
        return `CARPET ${dn[move.direction]} ×${move.roll_length}`;
      case MoveType.SEARCH:
        return `SEARCH (${move.search_loc?.[0]},${move.search_loc?.[1]})`;
      default:
        return '?';
    }
  }

  const API = { evaluate, Expectiminimax, rankMoves, moveLabel };
  if (typeof window !== 'undefined') window.MINIMAX = API;
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
})();
