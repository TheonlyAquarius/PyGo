"""Simple GTP engine wrapper using the Minimal MCTS.

Usage:
    python gtp_engine.py  # starts a stdin/stdout GTP loop

Supported commands (subset):
    - protocol_version
    - name
    - version
    - boardsize <N>
    - clear_board
    - play <color> <vertex>
    - genmove <color>
    - set_param playouts <N>
    - quit

The engine logs playout timing to stderr as ``# PLAYOUTS: <seconds>s``.
"""

import sys
import time

import torch

from mcts import MCTS # MinimalGoBoard is removed from mcts.py
from model import ModelConfig, KataGoModel
from board import GoBoard # Import GoBoard


LETTERS = "ABCDEFGHJKLMNOPQRST"


def coord_to_index(vertex: str, size: int) -> int:
    vertex = vertex.strip().lower()
    if vertex == "pass":
        return size * size
    if len(vertex) < 2:
        raise ValueError("invalid vertex")
    col = vertex[0].upper()
    row = vertex[1:]
    if col not in LETTERS[:size]:
        raise ValueError("invalid vertex")
    try:
        row = int(row)
    except ValueError as exc:
        raise ValueError("invalid vertex") from exc
    x = LETTERS.index(col)
    y = size - row
    if not (0 <= x < size and 0 <= y < size):
        raise ValueError("invalid vertex")
    return y * size + x


def index_to_coord(idx: int, size: int) -> str:
    if idx == size * size:
        return "pass"
    x = idx % size
    y = idx // size
    col = LETTERS[x]
    row = size - y
    return f"{col}{row}"


class GtpEngine:
    def __init__(self, mcts: MCTS):
        self.mcts = mcts
        self.komi: float = 7.5 # Default komi for the engine
        # Use GoBoard, mcts.config.board_size should be from ModelConfig
        self.board = GoBoard(mcts.config.board_size)
        self.board.komi = self.komi # Ensure new board gets engine's komi
        # self.mask is no longer needed
        self.implemented_commands = {
            "protocol_version",
            "name",
            "version",
            "clear_board",
            "boardsize",
            "play",
            "genmove",
            "quit",
            "set_param",
            "known_command",
            "list_commands",
            "komi",
            "get_komi"
        }

    # Utility responses --------------------------------------------------
    def respond(self, msg: str = "") -> None:
        sys.stdout.write(f"= {msg}\n")
        sys.stdout.flush()

    def error(self, msg: str) -> None:
        sys.stdout.write(f"? {msg}\n")
        sys.stdout.flush()

    # Command handlers ---------------------------------------------------
    def handle_play(self, color: str, vertex: str) -> None:
        try:
            idx = coord_to_index(vertex, self.board.size)
        except ValueError:
            self.error("invalid vertex")
            return
        # Set player turn correctly in GoBoard
        player_to_set = 1 if color.lower().startswith("b") else -1
        if self.board.to_play != player_to_set:
            self.board.to_play = player_to_set # Force turn if GTP says so, though ideally it matches

        try:
            # GoBoard.play_move will check legality (occupation, ko, suicide)
            self.board.play_move(idx)
            self.respond()
        except ValueError as e:
            self.error(f"illegal move: {e}")

    def handle_genmove(self, color: str) -> None:
        # Set player turn correctly in GoBoard
        player_to_set = 1 if color.lower().startswith("b") else -1
        if self.board.to_play != player_to_set:
             self.board.to_play = player_to_set # Ensure MCTS searches for the correct player

        start = time.time()
        # MCTS search no longer takes a mask
        best_move = self.mcts.search(self.board)
        elapsed = time.time() - start
        print(f"# PLAYOUTS: {elapsed:.2f}s ({self.mcts.n_playouts} playouts)", file=sys.stderr)

        try:
            vertex = index_to_coord(best_move, self.board.size)
            self.board.play_move(best_move) # Play the best move on our GoBoard
            self.respond(vertex)
        except ValueError as e: # Should not happen if MCTS returns valid move
            self.error(f"MCTS generated an illegal move: {best_move}, error: {e}")
        except Exception as e: # Catch any other unexpected errors
            self.error(f"Error processing MCTS move: {e}")


    def handle_boardsize(self, size: str) -> None:
        try:
            s = int(size)
            if s <= 0: # Basic validation for board size
                self.error("invalid boardsize: must be positive")
                return
        except ValueError:
            self.error("invalid boardsize: not an integer")
            return


        if s != self.mcts.config.board_size:
            self.error(f"board size {s}x{s} not supported, current model is {self.mcts.config.board_size}x{self.mcts.config.board_size}")
            return

        # If s is the same as current model config size, effectively clear_board
        self.board = GoBoard(s)
        self.board.komi = self.komi # Ensure new board gets engine's komi
        self.respond()

    def handle_set_param(self, name: str, value: str) -> None:
        if name != "playouts":
            self.error("unknown parameter")
            return
        try:
            n = int(value)
        except ValueError:
            self.error("invalid number")
            return
        self.mcts.n_playouts = n
        self.respond()

    # Main command dispatcher -------------------------------------------
    def handle_command(self, line: str) -> bool:
        if not line:
            return True
        parts = line.split()
        cmd = parts[0]
        args = parts[1:]
        if cmd == "quit":
            self.respond()
            return False
        elif cmd == "protocol_version":
            self.respond("2")
        elif cmd == "name":
            self.respond("PyGo")
        elif cmd == "version":
            self.respond("0.1")
        elif cmd == "clear_board":
            # Reinitialize GoBoard using the model's configured board size
            self.board = GoBoard(self.mcts.config.board_size)
            self.board.komi = self.komi # Apply current engine komi
            self.respond()
        elif cmd == "boardsize" and len(args) == 1:
            self.handle_boardsize(args[0])
        elif cmd == "play" and len(args) == 2:
            self.handle_play(args[0], args[1])
        elif cmd == "genmove" and len(args) == 1:
            self.handle_genmove(args[0])
        elif cmd == "set_param" and len(args) == 2:
            self.handle_set_param(args[0], args[1])
        elif cmd == "known_command":
            if len(args) == 1:
                if args[0] in self.implemented_commands:
                    self.respond("true")
                else:
                    self.respond("false")
            else:
                self.error("wrong number of arguments")
        elif cmd == "list_commands":
            if len(args) == 0:
                self.respond("\n".join(sorted(list(self.implemented_commands))))
            else:
                self.error("wrong number of arguments")
        elif cmd == "komi":
            if len(args) == 1:
                try:
                    new_komi = float(args[0])
                    self.komi = new_komi
                    self.board.komi = self.komi # Update board's komi as well
                    self.respond()
                except ValueError:
                    self.error("invalid komi value")
            else:
                self.error("wrong number of arguments")
        elif cmd == "get_komi":
            if len(args) == 0:
                self.respond(str(self.komi))
            else:
                self.error("wrong number of arguments")
        else:
            self.error("unknown command")
        return True


def main() -> None:
    config = ModelConfig()  # Use default config for internal model
    model = KataGoModel(config)
    mcts = MCTS(model, config) # Pass internal model and config
    engine = GtpEngine(mcts)
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            if not engine.handle_command(line.strip()):
                break
        except EOFError:
            break


if __name__ == "__main__":
    main()
