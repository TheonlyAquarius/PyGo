from typing import Optional, Set, Tuple, List, Dict
import torch

class GoBoard:
    """
    Represents a Go board and its state.
    """

    def __init__(self, size: int):
        if not isinstance(size, int) or size <= 0:
            raise ValueError("Board size must be a positive integer.")
        self.size: int = size
        self.board: torch.Tensor = torch.zeros((size, size), dtype=torch.int8)
        self.to_play: int = 1  # 1 for black, -1 for white
        self.history: List[Tuple[int, int]] = []
        self.ko_point: Optional[int] = None
        self.captured_stones: Dict[int, int] = {1: 0, -1: 0}
        self.PASS_MOVE: int = size * size
        self.komi: float = 7.5 # Default komi

    def _get_group(self, r_start: int, c_start: int) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        Finds the connected group of stones and their liberties.
        Uses Breadth-First Search (BFS).
        """
        if not (0 <= r_start < self.size and 0 <= c_start < self.size):
            return set(), set()

        player_color = self.board[r_start, c_start].item()
        if player_color == 0:
            return set(), set()

        group_stones: Set[Tuple[int, int]] = set()
        group_liberties: Set[Tuple[int, int]] = set()
        q: List[Tuple[int, int]] = [(r_start, c_start)]
        visited: Set[Tuple[int, int]] = {(r_start, c_start)}

        while q:
            r, c = q.pop(0)
            group_stones.add((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc

                if 0 <= nr < self.size and 0 <= nc < self.size:
                    neighbor_stone = self.board[nr, nc].item()
                    if neighbor_stone == 0:
                        group_liberties.add((nr, nc))
                    elif neighbor_stone == player_color and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return group_stones, group_liberties

    def count_liberties(self, r: int, c: int) -> int:
        """
        Counts the liberties of the group to which the stone at (r, c) belongs.
        """
        if not (0 <= r < self.size and 0 <= c < self.size) or self.board[r,c] == 0:
            return 0
        _, liberties = self._get_group(r, c)
        return len(liberties)

    def is_legal_move(self, move_index: int, player: Optional[int] = None) -> bool:
        """
        Checks if a move is legal.
        """
        current_player = player if player is not None else self.to_play

        if move_index == self.PASS_MOVE:
            return True

        if not (0 <= move_index < self.PASS_MOVE):
            return False

        r, c = move_index // self.size, move_index % self.size

        if self.board[r, c] != 0:
            return False  # Point is occupied

        if move_index == self.ko_point:
            return False  # Simple Ko

        # Check for suicide
        original_stone = self.board[r, c].item()
        self.board[r, c] = current_player
        _, liberties_of_played_stone = self._get_group(r,c)
        is_suicide = len(liberties_of_played_stone) == 0

        if is_suicide:
            captured_any = False
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == -current_player:
                    _, opp_liberties = self._get_group(nr, nc)
                    if len(opp_liberties) == 0:
                        captured_any = True
                        break
            if not captured_any:
                self.board[r, c] = original_stone # Revert
                return False # Suicide

        self.board[r, c] = original_stone # Revert temporary placement
        return True

    def play_move(self, move_index: int) -> None:
        """
        Plays a move on the board.
        """
        if not self.is_legal_move(move_index):
            r,c = (move_index // self.size, move_index % self.size) if move_index != self.PASS_MOVE else (-1,-1)
            raise ValueError(f"Illegal move: index {move_index} ({r},{c}) for player {self.to_play}")

        player = self.to_play
        self.ko_point = None

        if move_index == self.PASS_MOVE:
            self.history.append((player, move_index))
            self.to_play *= -1
            return

        r, c = move_index // self.size, move_index % self.size
        self.board[r, c] = player

        captured_stones_count_this_move = 0
        captured_single_stone_location: Optional[Tuple[int,int]] = None

        # Capture logic
        opponent = -player
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == opponent:
                group_stones, group_liberties = self._get_group(nr, nc)
                if not group_liberties:
                    if len(group_stones) == 1:
                        captured_single_stone_location = list(group_stones)[0]
                    for stone_r, stone_c in group_stones:
                        self.board[stone_r, stone_c] = 0
                        captured_stones_count_this_move += 1
        
        if captured_stones_count_this_move > 0:
            self.captured_stones[player] += captured_stones_count_this_move

        # Ko update
        if captured_stones_count_this_move == 1 and captured_single_stone_location is not None:
            _, played_stone_liberties = self._get_group(r,c)
            if len(played_stone_liberties) == 1:
                if list(played_stone_liberties)[0] == captured_single_stone_location:
                    self.ko_point = captured_single_stone_location[0] * self.size + captured_single_stone_location[1]

        self.history.append((player, move_index))
        self.to_play *= -1

    def get_board_state(self) -> torch.Tensor:
        """
        Returns a copy of the current board state.
        """
        return self.board.clone()

    def copy(self) -> 'GoBoard':
        """
        Creates a deep copy of the current board state.
        """
        new_board = GoBoard(self.size)
        new_board.board = self.board.clone()
        new_board.to_play = self.to_play
        new_board.history = self.history[:]
        new_board.ko_point = self.ko_point
        new_board.captured_stones = self.captured_stones.copy()
        new_board.komi = self.komi
        return new_board

    def legal_moves(self) -> List[int]:
        """
        Returns a list of all legal move indices for the current player.
        """
        moves = []
        for i in range(self.size * self.size):
            if self.is_legal_move(i):
                moves.append(i)
        if self.is_legal_move(self.PASS_MOVE):
            moves.append(self.PASS_MOVE)
        return moves

    def __str__(self) -> str:
        chars = {0: '.', 1: 'X', -1: 'O'}
        s = f"To play: {chars[self.to_play]} (Black: {self.captured_stones[1]}, White: {self.captured_stones[-1]}) Komi: {self.komi}\n"
        s += "  " + " ".join([chr(ord('A') + i + (i >= 8)) for i in range(self.size)]) + "\n"
        for r in range(self.size):
            s += f"{self.size - r:2d} " + " ".join([chars[self.board[r, c].item()] for c in range(self.size)]) + "\n"
        if self.ko_point is not None:
            kr, kc = self.ko_point // self.size, self.ko_point % self.size
            ko_char_col = chr(ord('A') + kc + (kc >= 8))
            s += f"Ko point: ({ko_char_col}{self.size-kr})\n"
        return s