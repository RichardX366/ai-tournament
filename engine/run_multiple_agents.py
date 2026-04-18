import math
import os
import pathlib
import random
import sys
import time
import multiprocessing

from board_utils import get_history_json
from game.board import Board
from game.enums import ResultArbiter
from gameplay import play_game


PAIRS = 32


def _extract_a_b_workers(board: Board):
    if board.player_worker.is_player_a:
        return board.player_worker, board.opponent_worker
    return board.opponent_worker, board.player_worker


def worker(player_a_name, player_b_name, play_directory, i, return_dict, seed=None):
    if seed is not None:
        random.seed(seed)

    final_board, rat_position_history, spawn_a, spawn_b, message_a, message_b = (
        play_game(
            play_directory,
            play_directory,
            player_a_name,
            player_b_name,
            display_game=False,
            delay=0,
            clear_screen=False,
            record=True,
            limit_resources=False,
        )
    )

    records_dir = os.path.join(play_directory, "matches")
    os.makedirs(records_dir, exist_ok=True)
    file_i = 0
    while True:
        out_file = f"{player_a_name}_{player_b_name}_{file_i}.json"
        out_path = os.path.join(records_dir, out_file)
        if not os.path.exists(out_path):
            break
        file_i += 1

    with open(out_path, "w") as fp:
        fp.write(
            get_history_json(
                final_board,
                rat_position_history,
                spawn_a,
                spawn_b,
                message_a,
                message_b,
            )
        )

    worker_a, worker_b = _extract_a_b_workers(final_board)
    return_dict[i] = {
        "result": ResultArbiter(final_board.get_winner()).name,
        "time_left_end": {
            "a": float(worker_a.time_left),
            "b": float(worker_b.time_left),
        },
        "scores": {
            "a": int(worker_a.get_points()),
            "b": int(worker_b.get_points()),
        },
    }


def _ttest_1samp(data):
    """One-sample t-test against mu=0. Returns (t, p, mean, ci_half_95)."""
    n = len(data)
    if n < 2:
        return 0.0, 1.0, (data[0] if n else 0.0), float("inf")

    mean = sum(data) / n
    var = sum((x - mean) ** 2 for x in data) / (n - 1)
    se = math.sqrt(var / n) if var > 0 else 0.0
    t = mean / se if se > 0 else 0.0

    try:
        from scipy import stats as _stats

        p = float(_stats.ttest_1samp(data, 0).pvalue)
        ci_half = float(_stats.t.ppf(0.975, df=n - 1)) * se
    except ImportError:
        _T95 = {
            1: 12.706,
            2: 4.303,
            3: 3.182,
            4: 2.776,
            5: 2.571,
            6: 2.447,
            7: 2.365,
            8: 2.306,
            9: 2.262,
            10: 2.228,
            15: 2.131,
            20: 2.086,
            25: 2.060,
            30: 2.042,
            40: 2.021,
            60: 2.000,
            120: 1.980,
        }
        best_df = min(_T95, key=lambda k: abs(k - (n - 1)))
        ci_half = _T95[best_df] * se
        p = 1.0 - math.erf(abs(t) / math.sqrt(2))  # normal approximation

    return t, p, mean, ci_half


def _needed_n(mean, std):
    """Pairs needed for 80% power at this observed effect size."""
    if std <= 0 or mean == 0:
        return float("inf")
    return math.ceil(((1.96 + 0.842) / (abs(mean) / std)) ** 2)


def run_paired_test(
    player_a_name: str, player_b_name: str, n_pairs: int = PAIRS, batch_size: int = 4
):
    """
    For each of n_pairs seeds, run A-vs-B and B-vs-A on identical board conditions.
    Paired margin for seed i:
        (a_score_in_AxB - b_score_in_AxB + a_score_in_BxA - b_score_in_BxA) / 2
    Runs a one-sample t-test on these margins against 0.
    """
    total_start = time.perf_counter()
    top_level = pathlib.Path(__file__).parent.parent.resolve()
    play_directory = os.path.join(top_level, "3600-agents")
    seeds = [i * 10 for i in range(n_pairs)]

    ab_results, ba_results = {}, {}

    pair_idx = 0
    while pair_idx < n_pairs:
        batch_seeds = seeds[pair_idx : pair_idx + batch_size]

        manager = multiprocessing.Manager()
        ab_dict = manager.dict()
        ba_dict = manager.dict()

        procs = []
        for local_i, seed in enumerate(batch_seeds):
            procs.append(
                multiprocessing.Process(
                    target=worker,
                    args=(
                        player_a_name,
                        player_b_name,
                        play_directory,
                        local_i,
                        ab_dict,
                        seed,
                    ),
                )
            )
            procs.append(
                multiprocessing.Process(
                    target=worker,
                    args=(
                        player_b_name,
                        player_a_name,
                        play_directory,
                        local_i,
                        ba_dict,
                        seed,
                    ),
                )
            )
        for p in procs:
            p.start()
        for p in procs:
            p.join()

        for local_i in range(len(batch_seeds)):
            ab_results[pair_idx + local_i] = ab_dict[local_i]
            ba_results[pair_idx + local_i] = ba_dict[local_i]

        end = min(pair_idx + batch_size, n_pairs)
        print(f"  Completed pairs {pair_idx + 1}–{end} / {n_pairs}")
        pair_idx += batch_size

    paired_margins = []
    wins_a = wins_b = ties = 0
    time_left_a, time_left_b = [], []

    for i in range(n_pairs):
        ab, ba = ab_results[i], ba_results[i]
        margin_ab = ab["scores"]["a"] - ab["scores"]["b"]
        margin_ba = ba["scores"]["b"] - ba["scores"]["a"]
        paired_margins.append((margin_ab + margin_ba) / 2.0)

        # player_a_name plays as "a" in AB, as "b" in BA
        time_left_a.append(ab["time_left_end"]["a"])
        time_left_a.append(ba["time_left_end"]["b"])
        time_left_b.append(ab["time_left_end"]["b"])
        time_left_b.append(ba["time_left_end"]["a"])

        for result, is_ab in [(ab["result"], True), (ba["result"], False)]:
            if result == ResultArbiter.PLAYER_A.name:
                wins_a += 1 if is_ab else 0
                wins_b += 0 if is_ab else 1
            elif result == ResultArbiter.PLAYER_B.name:
                wins_b += 1 if is_ab else 0
                wins_a += 0 if is_ab else 1
            else:
                ties += 1

    contested = wins_a + wins_b
    t, p, mean, ci = _ttest_1samp(paired_margins)
    std = math.sqrt(sum((x - mean) ** 2 for x in paired_margins) / max(1, n_pairs - 1))

    return {
        "player_a": player_a_name,
        "player_b": player_b_name,
        "n_pairs": n_pairs,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "win_rate_a": wins_a / contested if contested else 0.5,
        "mean_margin": mean,
        "ci_half_95": ci,
        "t_stat": t,
        "p_value": p,
        "needed_n": _needed_n(mean, std),
        "elapsed_s": time.perf_counter() - total_start,
        "paired_margins": paired_margins,
        "avg_time_left_a": sum(time_left_a) / len(time_left_a) if time_left_a else 0.0,
        "avg_time_left_b": sum(time_left_b) / len(time_left_b) if time_left_b else 0.0,
    }


def _print_report(r):
    a, b = r["player_a"], r["player_b"]
    print()
    print("=" * 60)
    print(f"  Paired AB Test: {a} vs {b}")
    print(f"  {r['n_pairs']} pairs · {r['n_pairs'] * 2} total games")
    print("=" * 60)
    print(
        f"  Win rate ({a}): {r['win_rate_a']:.1%}  ({r['wins_a']}W / {r['wins_b']}L / {r['ties']}T)"
    )
    lo = r["mean_margin"] - r["ci_half_95"]
    hi = r["mean_margin"] + r["ci_half_95"]
    print(
        f"  Mean score margin: {r['mean_margin']:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]"
    )
    print(f"  t = {r['t_stat']:.3f},  p = {r['p_value']:.4f}")
    if r["p_value"] < 0.05:
        winner = a if r["mean_margin"] > 0 else b
        print(f"\n  SIGNIFICANT (p < 0.05): {winner} is better.")
    else:
        print(f"\n  Not significant (p = {r['p_value']:.3f}).", end="  ")
        n = r["needed_n"]
        if n < 500:
            print(f"Need ~{n} pairs for 80% power.")
        else:
            print("Effect size too small to detect reliably.")
    print(f"\n  Pair margins: {[f'{m:+.1f}' for m in r['paired_margins']]}")
    print(f"  Avg time left — {a}: {r['avg_time_left_a']:.1f}s  |  {b}: {r['avg_time_left_b']:.1f}s")
    print(f"  Time: {r['elapsed_s']:.1f}s")
    print("=" * 60)
    print()


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <player_a> <player_b>")
        sys.exit(1)

    a, b = sys.argv[1], sys.argv[2]
    print(f"Paired AB test: {a} vs {b}  ({PAIRS} pairs, {PAIRS * 2} games)\n")
    _print_report(run_paired_test(a, b))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
