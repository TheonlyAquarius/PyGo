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

from mcts import MCTS, MinimalGoBoard, ModelConfig, KataGoModel


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
        self.board = MinimalGoBoard(mcts.config.board_size)
        self.mask = torch.ones(
            mcts.config.board_size, mcts.config.board_size, dtype=torch.float32
        )

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
        if color.lower().startswith("b"):
            expected = 1
        else:
            expected = -1
        if self.board.to_play != expected:
            self.board.to_play = expected
        if idx != self.board.size * self.board.size:
            x = idx % self.board.size
            y = idx // self.board.size
            if self.board.board[y, x] != 0:
                self.error("illegal move")
                return
        self.board.play(idx)
        self.respond()

    def handle_genmove(self, color: str) -> None:
        if color.lower().startswith("b"):
            self.board.to_play = 1
        else:
            self.board.to_play = -1
        start = time.time()
        best = self.mcts.search(self.board, self.mask)
        elapsed = time.time() - start
        print(f"# PLAYOUTS: {elapsed:.2f}s", file=sys.stderr)
        vertex = index_to_coord(best, self.board.size)
        self.board.play(best)
        self.respond(vertex)

    def handle_boardsize(self, size: str) -> None:
        try:
            s = int(size)
        except ValueError:
            self.error("invalid boardsize")
            return
        self.board = MinimalGoBoard(s)
        self.mask = torch.ones(s, s, dtype=torch.float32)
        self.mcts.config.board_size = s
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
            self.board = MinimalGoBoard(self.board.size)
            self.respond()
        elif cmd == "boardsize" and len(args) == 1:
            self.handle_boardsize(args[0])
        elif cmd == "play" and len(args) == 2:
            self.handle_play(args[0], args[1])
        elif cmd == "genmove" and len(args) == 1:
            self.handle_genmove(args[0])
        elif cmd == "set_param" and len(args) == 2:
            self.handle_set_param(args[0], args[1])
        else:
            self.error("unknown command")
        return True


def main() -> None:
    config = ModelConfig()
    model = KataGoModel(config)
    mcts = MCTS(model, config)
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
