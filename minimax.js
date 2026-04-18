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

  // --- Transposition table constants ------------------------------------
  const TT_EXACT = 0;
  const TT_LOWER = 1; // score is a lower bound (beta cutoff)
  const TT_UPPER = 2; // score is an upper bound (alpha cutoff)
  const TT_MAX_SIZE = 1 << 18; // 262144 entries

  function boardKey(board) {
    const p = board.player_worker;
    const o = board.opponent_worker;
    // Use a string key — fast and collision-free for Map lookups
    return `${board._primed_mask},${board._carpet_mask},${p.position[0]},${p.position[1]},${o.position[0]},${o.position[1]},${p.points},${o.points},${p.turns_left},${o.turns_left}`;
  }

  function moveKey(mv) {
    return `${mv.move_type},${mv.direction},${mv.roll_length || 0}`;
  }

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

  // --- Eval weights (mutable — updated by UI sliders) --------------------
  const weights = {
    bestCarpet: 0.5,
    lfTotal: 0.1,
    lfBest: 0.3,
    center: 0.1,
    prime: 0.1,
    openness: 0.1,
    ext: 0.2,
  };

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
      [
        [0, -1],
        [0, 1],
      ], // vertical: UP / DOWN
      [
        [-1, 0],
        [1, 0],
      ], // horizontal: LEFT / RIGHT
    ];

    for (const [[dx1, dy1], [dx2, dy2]] of axes) {
      let count1 = 0;
      let cx = x + dx1,
        cy = y + dy1;
      while (cx >= 0 && cx < 8 && cy >= 0 && cy < 8) {
        const bit = bitAt(cx, cy);
        if (
          (occupied & bit) !== 0n ||
          (cx === px && cy === py) ||
          (cx === ox && cy === oy)
        )
          break;
        // opponent can reach this tile before us — line is contested from here
        if (Math.abs(cx - ox) + Math.abs(cy - oy) < count1 + 1) break;
        count1++;
        cx += dx1;
        cy += dy1;
      }

      let count2 = 0;
      cx = x + dx2;
      cy = y + dy2;
      while (cx >= 0 && cx < 8 && cy >= 0 && cy < 8) {
        const bit = bitAt(cx, cy);
        if (
          (occupied & bit) !== 0n ||
          (cx === px && cy === py) ||
          (cx === ox && cy === oy)
        )
          break;
        // opponent can reach this tile before us — line is contested from here
        if (Math.abs(cx - ox) + Math.abs(cy - oy) < count2 + 1) break;
        count2++;
        cx += dx2;
        cy += dy2;
      }

      const selfOk = (occupied & bitAt(x, y)) !== 0n ? 0 : 1;
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
    if (((primed | carpet | blocked) & bit) !== 0n) return 0;

    let best = 0;
    for (const dir of DIRECTIONS) {
      const [dx, dy] = DIR_MOVEMENTS[dir];
      const nx = x + dx,
        ny = y + dy;
      if (!(nx >= 0 && nx < 8 && ny >= 0 && ny < 8)) continue;
      const nextBit = bitAt(nx, ny);
      if (((blocked | primed) & nextBit) !== 0n) continue;
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
      while (
        bx >= 0 &&
        bx < 8 &&
        by >= 0 &&
        by < 8 &&
        (primed & bitAt(bx, by)) !== 0n
      ) {
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
        if ((space & ab) !== 0n && ((primed | carpet | blocked) & ab) === 0n) {
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

  // --- Primed run ownership ------------------------------------------------
  // Each horizontal/vertical run of primed tiles (length >= 2) is treated as
  // one indivisible unit.  A run is claimed by whichever agent has the smaller
  // Manhattan distance to either of its (non-blocked) endpoints — player wins
  // ties.  Runs are processed highest-value first; each side's first claim is
  // their best.  No two claims from different sides can come from the same run.
  // Returns [playerBest, opponentBest].
  function bestCarpetForSides(board, playerLoc, oppLoc) {
    const primed = board._primed_mask;
    const blocked = board._blocked_mask;
    if (primed === 0n) return [0, 0];

    const runs = []; // { ep1: [x,y]|null, ep2: [x,y]|null, val }

    function addRun(x1, y1, x2, y2, val) {
      const inB1 = x1 >= 0 && x1 < 8 && y1 >= 0 && y1 < 8;
      const inB2 = x2 >= 0 && x2 < 8 && y2 >= 0 && y2 < 8;
      const b1 = !inB1 || (blocked & bitAt(x1, y1)) !== 0n;
      const b2 = !inB2 || (blocked & bitAt(x2, y2)) !== 0n;
      if (!b1 || !b2)
        runs.push({
          ep1: b1 ? null : [x1, y1],
          ep2: b2 ? null : [x2, y2],
          val,
        });
    }

    // Horizontal runs — endpoints are the squares just outside the run
    for (let y = 0; y < 8; y++) {
      let rs = 0,
        rl = 0;
      for (let x = 0; x <= 8; x++) {
        const has = x < 8 && (primed & bitAt(x, y)) !== 0n;
        if (has) {
          if (rl === 0) rs = x;
          rl++;
        } else if (rl >= 2) {
          addRun(rs - 1, y, x, y, CARPET_PTS[Math.min(rl, 7)]);
          rl = 0;
        } else {
          rl = 0;
        }
      }
    }

    // Vertical runs — endpoints are the squares just outside the run
    for (let x = 0; x < 8; x++) {
      let rs = 0,
        rl = 0;
      for (let y = 0; y <= 8; y++) {
        const has = y < 8 && (primed & bitAt(x, y)) !== 0n;
        if (has) {
          if (rl === 0) rs = y;
          rl++;
        } else if (rl >= 2) {
          addRun(x, rs - 1, x, y, CARPET_PTS[Math.min(rl, 7)]);
          rl = 0;
        } else {
          rl = 0;
        }
      }
    }

    if (!runs.length) return [0, 0];

    runs.sort((a, b) => b.val - a.val);

    const [px, py] = playerLoc;
    const [ox, oy] = oppLoc;

    function distToRun(wx, wy, { ep1, ep2 }) {
      let d = Infinity;
      if (ep1) d = Math.min(d, Math.abs(ep1[0] - wx) + Math.abs(ep1[1] - wy));
      if (ep2) d = Math.min(d, Math.abs(ep2[0] - wx) + Math.abs(ep2[1] - wy));
      return d;
    }

    // Runs are sorted by raw value for claiming priority.
    // The returned score is distance-weighted: val * exp(-dist / 4),
    // so carpets the player is right next to score near their full value
    // while distant carpets contribute much less.
    let playerBest = 0,
      opponentBest = 0;
    for (const run of runs) {
      const dp = distToRun(px, py, run);
      const dop = distToRun(ox, oy, run);
      if (dp <= dop) {
        if (playerBest === 0) playerBest = run.val * Math.exp(-dp / 4);
      } else {
        if (opponentBest === 0) opponentBest = run.val * Math.exp(-dop / 4);
      }
      if (playerBest > 0 && opponentBest > 0) break;
    }

    return [playerBest, opponentBest];
  }

  // --- Static evaluation -------------------------------------------------
  function evaluate(board, logComponents = false) {
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

    const tc = board.turn_count;
    if (tc > 77) return scoreDiff;

    const playerMoves = board.get_valid_moves(false, true);
    const opponentMoves = board.get_valid_moves(true, true);

    function countPrimes(moves) {
      let n = 0;
      for (const mv of moves) if (mv.move_type === MoveType.PRIME) n++;
      return n;
    }
    const pPrime = countPrimes(playerMoves);
    const oPrime = countPrimes(opponentMoves);
    const [pBest, oBest] = bestCarpetForSides(board, [px, py], [ox, oy]);

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
        (blocked & bitAt(nx, ny)) === 0n
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
            (blocked & bitAt(nx2, ny2)) === 0n
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
        (blocked & bitAt(nx, ny)) === 0n
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
            (blocked & bitAt(nx2, ny2)) === 0n
          ) {
            oppOpen += 1;
          }
        }
      }
    }

    // Mid-game multiplier: ramp in 10→20, full 20→55, ramp out 55→65
    let midgameWeight;
    if (tc < 10) midgameWeight = 0;
    else if (tc < 20) midgameWeight = (tc - 10) / 10;
    else if (tc < 55) midgameWeight = 1;
    else if (tc < 65) midgameWeight = (65 - tc) / 10;
    else midgameWeight = 0;

    let midgameFactors = 0;
    if (midgameWeight > 0) {
      const playerCenter = 7 - (Math.abs(px - 3.5) + Math.abs(py - 3.5));
      const oppCenter = 7 - (Math.abs(ox - 3.5) + Math.abs(oy - 3.5));
      const playerExt = ext(board, [px, py], [px, py], [ox, oy]);
      const oppExt = ext(board, [ox, oy], [px, py], [ox, oy]);
      const pLF = lineFreedom(board, [px, py], [px, py], [ox, oy]);
      const oLF = lineFreedom(board, [ox, oy], [px, py], [ox, oy]);
      midgameFactors =
        midgameWeight *
        (weights.lfTotal * (pLF.total - oLF.total) +
          weights.lfBest * (pLF.bestRunway - oLF.bestRunway) +
          weights.center * (playerCenter - oppCenter) +
          weights.prime * (pPrime - oPrime) +
          weights.openness * (playerOpen - oppOpen) +
          weights.ext * (playerExt - oppExt));
      if (logComponents) {
        console.log('Midgame eval components:');
        console.log('  Line freedom total:', pLF.total, 'vs', oLF.total);
        console.log(
          '  Line freedom best runway:',
          pLF.bestRunway,
          'vs',
          oLF.bestRunway,
        );
        console.log('  Center proximity:', playerCenter, 'vs', oppCenter);
        console.log('  Prime count:', pPrime, 'vs', oPrime);
        console.log('  Openness:', playerOpen, 'vs', oppOpen);
        console.log(
          '  Extension value:',
          playerExt.toFixed(2),
          'vs',
          oppExt.toFixed(2),
        );
      }
    }

    if (logComponents) {
      console.log('Score diff:', scoreDiff);
      console.log('Best carpet:', pBest.toFixed(2), 'vs', oBest.toFixed(2));
    }

    return scoreDiff + weights.bestCarpet * (pBest - oBest) + midgameFactors;
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
        if (
          bx >= 0 &&
          bx < 8 &&
          by >= 0 &&
          by < 8 &&
          (primed & bitAt(bx, by)) !== 0n
        ) {
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
      this._ttHits = 0;
      this._deadline = Infinity;
      this._tt = new Map(); // boardKey → { score, depth, flag, bestMk }
    }

    _now() {
      return typeof performance !== 'undefined'
        ? performance.now()
        : Date.now();
    }

    _negamax(board, depth, alpha, beta) {
      this._nodes++;
      if (depth <= 0 || board.is_game_over() || this._now() >= this._deadline) {
        return evaluate(board);
      }

      // ── TT probe ──────────────────────────────────────────────────
      const key = boardKey(board);
      const ttEntry = this._tt.get(key);
      let ttBestMk = null;

      if (ttEntry !== undefined) {
        if (ttEntry.depth >= depth) {
          if (ttEntry.flag === TT_EXACT) {
            this._ttHits++;
            return ttEntry.score;
          }
          if (ttEntry.flag === TT_LOWER && ttEntry.score >= beta) {
            this._ttHits++;
            return ttEntry.score;
          }
          if (ttEntry.flag === TT_UPPER && ttEntry.score <= alpha) {
            this._ttHits++;
            return ttEntry.score;
          }
        }
        ttBestMk = ttEntry.bestMk;
      }

      // ── Generate & order moves ────────────────────────────────────
      const moves = board.get_valid_moves(false, true);
      if (!moves.length) return evaluate(board);
      const ordered = orderMovesFast(board, moves);

      // Promote TT best move to front
      if (ttBestMk !== null) {
        for (let i = 0; i < ordered.length; i++) {
          if (moveKey(ordered[i]) === ttBestMk) {
            if (i > 0) ordered.unshift(ordered.splice(i, 1)[0]);
            break;
          }
        }
      }

      // ── Search ────────────────────────────────────────────────────
      const origAlpha = alpha;
      let best = -INF;
      let bestMk = ordered.length ? moveKey(ordered[0]) : null;

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
        if (val > best) {
          best = val;
          bestMk = moveKey(mv);
        }
        if (val > alpha) alpha = val;
        if (alpha >= beta) break;
      }

      // ── TT store ──────────────────────────────────────────────────
      if (best > -1e17) {
        let flag;
        if (best <= origAlpha) flag = TT_UPPER;
        else if (best >= beta) flag = TT_LOWER;
        else flag = TT_EXACT;

        if (ttEntry === undefined || depth >= ttEntry.depth) {
          this._tt.set(key, { score: best, depth, flag, bestMk });
        }
      }

      return best > -1e17 ? best : evaluate(board);
    }
  }

  // --- Move points (actual game points from a move) -------------------------
  function movePoints(mv) {
    if (mv.move_type === MoveType.CARPET)
      return CARPET_POINTS_TABLE[mv.roll_length] ?? 0;
    if (mv.move_type === MoveType.PRIME) return 1;
    return 0;
  }

  // --- PV-returning negamax -----------------------------------------------
  // Returns { value, pv: [ { label, side, pts } , ... ] }
  // Each pv entry records the move label, which side played it, and the
  // immediate game points scored by that move.
  // tt is a shared Map for transposition table lookups (move ordering only
  // in PV mode — we never cut short because we need the full PV).
  function negamaxPV(board, depth, alpha, beta, deadline, ply, tt) {
    const now =
      typeof performance !== 'undefined' ? performance.now() : Date.now();
    if (depth <= 0 || board.is_game_over() || now >= deadline) {
      return { value: evaluate(board), pv: [] };
    }
    const moves = board.get_valid_moves(false, true);
    if (!moves.length) return { value: evaluate(board), pv: [] };
    const ordered = orderMovesFast(board, moves);

    // Use TT for move ordering (but not cutoffs — we need the PV)
    const key = boardKey(board);
    const ttEntry = tt ? tt.get(key) : undefined;
    if (ttEntry !== undefined && ttEntry.bestMk !== null) {
      for (let i = 0; i < ordered.length; i++) {
        if (moveKey(ordered[i]) === ttEntry.bestMk) {
          if (i > 0) ordered.unshift(ordered.splice(i, 1)[0]);
          break;
        }
      }
    }

    let best = -INF;
    let bestPV = [];
    let bestMk = ordered.length ? moveKey(ordered[0]) : null;
    const origAlpha = alpha;

    for (let i = 0; i < ordered.length; i++) {
      if (
        (typeof performance !== 'undefined' ? performance.now() : Date.now()) >=
        deadline
      )
        break;
      const mv = ordered[i];
      const child = board.forecast_move(mv, true);
      if (!child) continue;
      child.reverse_perspective();

      const res = negamaxPV(
        child,
        depth - 1,
        -beta,
        -alpha,
        deadline,
        ply + 1,
        tt,
      );
      const val = -res.value;

      if (val > best) {
        best = val;
        bestMk = moveKey(mv);
        bestPV = [
          {
            label: moveLabel(mv),
            side: ply % 2,
            pts: movePoints(mv),
            move: mv,
          },
          ...res.pv,
        ];
      }
      if (val > alpha) alpha = val;
      if (alpha >= beta) break;
    }

    // Store in TT for move ordering benefit in subsequent calls
    if (tt && best > -1e17) {
      let flag;
      if (best <= origAlpha) flag = TT_UPPER;
      else if (best >= beta) flag = TT_LOWER;
      else flag = TT_EXACT;
      if (ttEntry === undefined || depth >= ttEntry.depth) {
        tt.set(key, { score: best, depth, flag, bestMk });
      }
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

    // Cap depth at remaining game turns
    const turnsRemaining =
      board.player_worker.turns_left + board.opponent_worker.turns_left;
    const effectiveMax = Math.min(maxDepth, Math.max(1, turnsRemaining - 1));

    const now =
      typeof performance !== 'undefined' ? performance.now() : Date.now();
    const deadline = now + timeBudget;

    // Shared TT across all root move searches for move ordering benefit
    const tt = new Map();

    const results = [];
    for (const mv of moves) {
      const child = board.forecast_move(mv, true);
      if (!child) continue;
      child.reverse_perspective();
      const res = negamaxPV(
        child,
        effectiveMax - 1,
        -INF,
        INF,
        deadline,
        1,
        tt,
      );
      const val = -res.value;
      results.push({
        move: mv,
        score: val - evaluate(board),
        label: moveLabel(mv),
        pv: [
          { label: moveLabel(mv), side: 0, pts: movePoints(mv), move: mv },
          ...res.pv,
        ],
      });
    }

    // Simulate Skipping Turn
    const child = board.get_copy();
    child.end_turn();
    child.reverse_perspective();
    const res = negamaxPV(child, effectiveMax - 1, -INF, INF, deadline, 1, tt);
    const val = -res.value;
    results.push({
      move: Game.Move.search(),
      score: val - evaluate(board),
      label: 'Search (Skip Turn)',
      pv: [
        {
          label: 'Search (Skip Turn)',
          side: 0,
          pts: 0,
          move: Game.Move.search(),
        },
        ...res.pv,
      ],
    });

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

  const API = { evaluate, Expectiminimax, rankMoves, moveLabel, weights };
  if (typeof window !== 'undefined') window.MINIMAX = API;
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
})();
