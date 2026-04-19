import importlib
import os
import pathlib
import sys
import time
import multiprocessing

from board_utils import from_board_array


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <player_name>")
        sys.exit(1)

    sim_time = time.perf_counter()

    top_level = pathlib.Path(__file__).parent.parent.resolve()
    play_directory = os.path.join(top_level, "3600-agents")
    sys.path.append(play_directory)

    player_name = sys.argv[1]

    importlib.import_module(player_name)
    module = importlib.import_module(player_name + ".agent")

    board = from_board_array(
        [
            ["B", "B", "C", "C", "E", "B", "B", "B"],
            ["B", "B", "C", "C", "P", "B", "B", "B"],
            ["C", "C", "C", "C", "C", "C", "P", "P"],
            ["C", "C", "C", "C", "C", "E", "E", "P"],
            ["C", "C", "C", "C", "C", "E", "E", "P"],
            ["C", "C", "C", "C", "C", "E", "P", "P"],
            ["B", "B", "E", "E", "E", "B", "B", "B"],
            ["B", "B", "E", "E", "E", "B", "B", "B"],
        ],
        (5, 2),
        (2, 5),
    )

    player = module.PlayerAgent(board)

    print(player.play(board, (0, 0), lambda: 999))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
