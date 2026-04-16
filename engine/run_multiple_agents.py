import os
import pathlib
import sys
import time
import multiprocessing

from board_utils import get_history_json
from game.board import Board
from game.enums import ResultArbiter
from gameplay import play_game


SIMULATIONS = 8


def _extract_a_b_workers(board: Board):
    """Return (worker_a, worker_b) regardless of current board perspective."""
    if board.player_worker.is_player_a:
        return board.player_worker, board.opponent_worker
    return board.opponent_worker, board.player_worker


def worker(player_a_name, player_b_name, play_directory, i, return_dict):
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


def run_simulations(
    player_a_name: str, player_b_name: str, simulations: int = SIMULATIONS
):
    sim_time = time.perf_counter()

    top_level = pathlib.Path(__file__).parent.parent.resolve()
    play_directory = os.path.join(top_level, "3600-agents")

    processes: list[multiprocessing.Process] = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    for i in range(simulations):
        p = multiprocessing.Process(
            target=worker,
            args=(player_a_name, player_b_name, play_directory, i, return_dict),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    matches = [return_dict[i] for i in range(simulations)]

    result_counts = {}
    time_left_a = 0.0
    time_left_b = 0.0
    score_a = 0
    score_b = 0

    for match in matches:
        result = match["result"]
        result_counts[result] = result_counts.get(result, 0) + 1
        time_left_a += match["time_left_end"]["a"]
        time_left_b += match["time_left_end"]["b"]
        score_a += match["scores"]["a"]
        score_b += match["scores"]["b"]

    avg_time_left = {
        "a": time_left_a / simulations if simulations else 0.0,
        "b": time_left_b / simulations if simulations else 0.0,
    }
    avg_scores = {
        "a": score_a / simulations if simulations else 0.0,
        "b": score_b / simulations if simulations else 0.0,
    }

    summary = {
        "player_a": player_a_name,
        "player_b": player_b_name,
        "simulations": simulations,
        "result_counts": result_counts,
        "avg_time_left_end": avg_time_left,
        "scores": {
            "avg_end": avg_scores,
        },
        "matches": matches,
        "simulation_time_s": time.perf_counter() - sim_time,
    }

    return summary


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python3 {sys.argv[0]} <player_a_name> <player_b_name>")
        sys.exit(1)

    summary = run_simulations(sys.argv[1], sys.argv[2], SIMULATIONS)
    print(f"Result counts: {summary['result_counts']}")
    print(
        "Avg time left at end (s):",
        f"A={summary['avg_time_left_end']['a']:.2f}",
        f"B={summary['avg_time_left_end']['b']:.2f}",
    )
    print(
        "Avg end score:",
        f"A={summary['scores']['avg_end']['a']:.2f}",
        f"B={summary['scores']['avg_end']['b']:.2f}",
    )
    print("Total simulation time:", summary["simulation_time_s"])


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
