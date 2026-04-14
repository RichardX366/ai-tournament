import os
import pathlib
import sys
import time
import multiprocessing

from board_utils import get_history_json
from game.board import Board
from game.enums import ResultArbiter
from gameplay import play_game


def worker(player_a_name, player_b_name, play_directory, i, return_dict):
    final_board, rat_position_history, spawn_a, spawn_b, message_a, message_b = (
        play_game(
            play_directory,
            play_directory,
            player_a_name,
            player_b_name,
            display_game=False,
            delay=0.0,
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

    return_dict[i] = ResultArbiter(final_board.get_winner()).name


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python3 {sys.argv[0]} <player_a_name> <player_b_name>")
        sys.exit(1)

    sim_time = time.perf_counter()

    top_level = pathlib.Path(__file__).parent.parent.resolve()
    play_directory = os.path.join(top_level, "3600-agents")

    player_a_name = sys.argv[1]
    player_b_name = sys.argv[2]

    results = []
    processes: list[multiprocessing.Process] = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    for i in range(16):
        p = multiprocessing.Process(
            target=worker,
            args=(player_a_name, player_b_name, play_directory, i, return_dict),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    for i in range(16):
        results.append(return_dict[i])

    result_counts = {}
    for result in results:
        result_counts[result] = result_counts.get(result, 0) + 1
    print(f"Result counts: {result_counts}")
    print("Total simulation time:", time.perf_counter() - sim_time)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
