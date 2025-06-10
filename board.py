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

        Args:
            r_start (int): Row of the starting stone.
            c_start (int): Column of the starting stone.

        Returns:
            Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
                A tuple containing:
                - group_stones: Set of (row, col) tuples for stones in the group.
                - group_liberties: Set of (row, col) tuples for liberties of the group.
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
                # No need to check out of bounds for liberties, as they must be on board

        return group_stones, group_liberties

    def count_liberties(self, r: int, c: int) -> int:
        """
        Counts the liberties of the group to which the stone at (r, c) belongs.
        Returns 0 if (r,c) is empty or out of bounds.
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

        if not (0 <= move_index < self.PASS_MOVE): # Check if move_index is within board bounds
            return False

        r, c = move_index // self.size, move_index % self.size

        if not (0 <= r < self.size and 0 <= c < self.size): # Should be redundant due to above check
            return False # Out of bounds

        if self.board[r, c] != 0:
            return False  # Point is occupied

        if move_index == self.ko_point:
            return False  # Simple Ko

        # Check for suicide
        # Temporarily place the stone
        original_stone = self.board[r, c].item() # Should be 0
        self.board[r, c] = current_player

        # Check if the placed stone itself has 0 liberties
        _, liberties_of_played_stone = self._get_group(r,c)

        is_suicide = len(liberties_of_played_stone) == 0

        if is_suicide:
            # If it has 0 liberties, check if this move captures any opponent stones.
            # If it captures, it's not suicide.
            opponent_player = -current_player
            captured_any = False
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == opponent_player:
                    # Check liberties of this specific opponent group *after* our stone is placed
                    _, opp_liberties = self._get_group(nr, nc)
                    if len(opp_liberties) == 0:
                        captured_any = True
                        break
            if not captured_any:
                self.board[r, c] = original_stone # Revert
                return False # Suicide

        # Revert temporary placement for the check
        self.board[r, c] = original_stone
        return True

    def play_move(self, move_index: int) -> None:
        """
        Plays a move on the board.
        Raises ValueError if the move is illegal.
        """
        if not self.is_legal_move(move_index):
            r,c = (move_index // self.size, move_index % self.size) if move_index != self.PASS_MOVE else (-1,-1)
            raise ValueError(f"Illegal move: index {move_index} ({r},{c}) for player {self.to_play}. Ko: {self.ko_point}, Board:\n{self.board}")

        player = self.to_play

        # Reset ko point unless a new ko is formed later
        # This must happen *before* checking captures that might clear the ko condition
        # but *after* the is_legal_move check for the current ko_point.
        # The actual ko point for next turn is determined at the end of this function.
        # For now, we assume no ko unless one is explicitly set.
        # This local variable `old_ko_point` is not strictly needed now but can be useful for debugging.
        # old_ko_point = self.ko_point
        self.ko_point = None


        if move_index == self.PASS_MOVE:
            self.history.append((player, move_index))
            self.to_play *= -1
            # self.ko_point = None # Already done above
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
                if not group_liberties: # No liberties, group is captured
                    if len(group_stones) == 1:
                        captured_single_stone_location = list(group_stones)[0]

                    for stone_r, stone_c in group_stones:
                        self.board[stone_r, stone_c] = 0
                        captured_stones_count_this_move += 1

        if captured_stones_count_this_move > 0:
            self.captured_stones[player] += captured_stones_count_this_move

        # Ko update
        if captured_stones_count_this_move == 1 and captured_single_stone_location is not None:
            # Check if the placed stone has only one liberty and that liberty is the captured stone's location
            # This means the stone just placed at (r,c) is now part of a group.
            # We need the liberties of the group containing (r,c)
            played_stone_group, played_stone_liberties = self._get_group(r,c)
            if len(played_stone_liberties) == 1:
                single_liberty_loc = list(played_stone_liberties)[0]
                if single_liberty_loc == captured_single_stone_location:
                    # Check if playing at captured_single_stone_location by opponent would be suicide
                    # (i.e. it would have 0 liberties itself without capturing the stone at (r,c))
                    # This is a simplified ko rule. A more precise one might check if the board state reverts.
                    self.ko_point = captured_single_stone_location[0] * self.size + captured_single_stone_location[1]
        # else: # self.ko_point is already None if not a ko

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
        new_board.history = self.history[:] # Shallow copy of list of tuples is fine
        new_board.ko_point = self.ko_point
        new_board.captured_stones = self.captured_stones.copy()
        new_board.komi = self.komi # Copy komi
        # PASS_MOVE is set by __init__ based on size, so no need to copy explicitly
        return new_board

    def legal_moves(self) -> List[int]:
        """
        Returns a list of all legal move indices for the current player.
        Includes the pass move.
        """
        moves = []
        for i in range(self.size * self.size): # Board points
            if self.is_legal_move(i):
                moves.append(i)
        if self.is_legal_move(self.PASS_MOVE): # Pass move
            moves.append(self.PASS_MOVE)
        return moves

    def __str__(self) -> str:
        chars = {0: '.', 1: 'X', -1: 'O'}
        s = f"To play: {chars[self.to_play]} (Black: {self.captured_stones[1]}, White: {self.captured_stones[-1]}) Komi: {self.komi}\n"
        s += "  " + " ".join([chr(ord('A') + i + (i >= 8)) for i in range(self.size)]) + "\n" # Skip 'I'
        for r in range(self.size):
            s += f"{self.size - r:2d} " + " ".join([chars[self.board[r, c].item()] for c in range(self.size)]) + "\n"
        if self.ko_point is not None:
            kr, kc = self.ko_point // self.size, self.ko_point % self.size
            ko_char_col = chr(ord('A') + kc + (kc >= 8)) # Skip 'I'
            s += f"Ko point: ({ko_char_col}{self.size-kr})\n"
        return s

if __name__ == '__main__':
    # Example Usage (for testing during development)
    board = GoBoard(5)
    print(board)

    # B plays C3 (index for 5x5: (5-3)*5 + 2 = 2*5 + 2 = 12)
    # A B C D E
    # 5 . . . . .
    # 4 . . . . .
    # 3 . . X . .  (X is at (2,2) in 0-indexed tensor)
    # 2 . . . . .
    # 1 . . . . .
    # board.play_move(2*5 + 2) # B at C3
    # print(board)
    #
    # # W plays C4 (index for 5x5: (5-4)*5 + 2 = 1*5 + 2 = 7)
    # # A B C D E
    # # 5 . . . . .
    # # 4 . . O . .  (O is at (1,2) in 0-indexed tensor)
    # # 3 . . X . .
    # # 2 . . . . .
    # # 1 . . . . .
    # board.play_move(1*5 + 2) # W at C4
    # print(board)

    # Test basic capture
    # B: D3 (2,3), W: C3 (2,2), B: D2 (3,3), W: E3 (2,4), B: C2 (3,2), W: D4 (1,3), B: E2 (3,4)
    # W plays D3 (2,3), capturing nothing
    # B plays C3 (2,2)
    # W plays D2 (3,3)
    # B plays E3 (2,4)
    # W plays C2 (3,2)
    # B plays D4 (1,3)
    # W plays E2 (3,4)
    # B plays D3 (2,3) to capture C3, D2, E3, C2, D4, E2
    # This setup is for a 9x9 board typically. Let's try a simpler 5x5 capture.
    # . . . . .
    # . O X . .  X at (1,2), O at (1,1)
    # . X . . .  X at (2,1)
    # . . . . .
    # . . . . .
    # Black plays (1,1) to capture (1,2)
    # board = GoBoard(5)
    # board.play_move(1*5+2) # B at (1,2) - C4
    # print(board)
    # board.play_move(2*5+1) # W at (2,1) - B3
    # print(board)
    # board.play_move(1*5+1) # B at (1,1) - B4, to capture (1,2)
    # print(board) # This move is suicide without capture, should be illegal if suicide rule is on
    # board.play_move(0*5+2) # W at (0,2) - C5
    # print(board)
    # board.play_move(2*5+2) # B at (2,2) - C3
    # print(board)
    # # Now board is:
    # # . . O . . (0,2) W
    # # . X X . . (1,1) B, (1,2) B
    # # . O X . . (2,1) W, (2,2) B
    # # If W plays (1,0) (A4), it should capture (1,1)B
    # board.play_move(1*5+0) # W at (1,0)
    # print(board)
    # print(f"Captured by White: {board.captured_stones[-1]}")
    # print(f"Captured by Black: {board.captured_stones[1]}")

    # Test Ko
    # . . . . .
    # . X O . .  X at (1,1), O at (1,2)
    # X O . . .  X at (2,0), O at (2,1)
    # . . . . .
    g = GoBoard(5)
    g.play_move(g.coords_to_idx(1,1)) # B plays (1,1) (B4)
    g.play_move(g.coords_to_idx(1,2)) # W plays (1,2) (C4)
    g.play_move(g.coords_to_idx(2,0)) # B plays (2,0) (A3)
    g.play_move(g.coords_to_idx(2,1)) # W plays (2,1) (B3)
    print(g)
    # Current board:
    # . . . . .
    # . X O . .
    # X O . . .
    # . . . . .
    # B plays (2,2) C3. This should capture (1,2)O and (2,1)O if set up like this
    # Let's make a simpler ko:
    # . O X .
    # O X . X
    # . O X .
    # . . . .
    # B plays X at (0,2). W plays O at (0,1)
    # B plays X at (1,1). W plays O at (1,0)
    # B plays X at (2,2). W plays O at (2,1)
    # B plays X at (1,3)

    # Setup for Ko:
    #   A B C D
    # 4 . O X .   (0,1) (0,2)
    # 3 O X . .   (1,0) (1,1)
    # 2 . O . .   (2,1)
    # 1 . . . .
    board = GoBoard(4)
    # Helper for tests
    def c2i(r,c,s=board.size): return r*s+c
    board.to_play=1 #B
    board.play_move(c2i(0,2)) # B X at (0,2)
    board.play_move(c2i(0,1)) # W O at (0,1)
    board.play_move(c2i(1,1)) # B X at (1,1)
    board.play_move(c2i(1,0)) # W O at (1,0)
    board.play_move(c2i(2,1)) # B X at (2,1) -> This should be B, Mistake in manual trace, should be W
    # Corrected Ko setup:
    #   A B C D E
    # 5 . . . . .
    # 4 . X O . .  (B4, C4)
    # 3 X O . . .  (A3, B3)
    # 2 . X O . .  (B2, C2)
    # 1 . . . . .
    # B plays B4 (1,1), W plays C4 (1,2)
    # B plays A3 (2,0), W plays B3 (2,1)
    # B plays B2 (3,1), W plays C2 (3,2)
    # B plays C3 (2,2) - captures B3(O at 2,1)
    # W plays B3 (2,1) - captures C3(X at 2,2) -> KO

    ko_board = GoBoard(size=5)
    ko_board.play_move(ko_board.coords_to_idx(1,1)) # B: B4
    ko_board.play_move(ko_board.coords_to_idx(1,2)) # W: C4
    ko_board.play_move(ko_board.coords_to_idx(2,0)) # B: A3
    ko_board.play_move(ko_board.coords_to_idx(2,1)) # W: B3 (O)
    ko_board.play_move(ko_board.coords_to_idx(3,1)) # B: B2
    ko_board.play_move(ko_board.coords_to_idx(3,2)) # W: C2
    print("Initial Ko Setup:")
    print(ko_board)
    # B plays C3 (idx 2,2 for 5x5 is 2*5+2=12)
    # This captures W's stone at B3 (idx 2,1 for 5x5 is 2*5+1=11)
    print("B plays C3 (2,2), capturing W at B3 (2,1)")
    ko_board.play_move(ko_board.coords_to_idx(2,2)) # B: C3
    print(ko_board)
    print(f"Ko point: {ko_board.ko_point} (should be index of B3: {ko_board.coords_to_idx(2,1)})")

    print(f"Is W playing B3 (2,1) legal? {ko_board.is_legal_move(ko_board.coords_to_idx(2,1))}") # Should be False (Ko)

    # Make a non-ko move for W
    print("W makes a non-ko move (A5)")
    ko_board.play_move(ko_board.coords_to_idx(0,0)) # W: A5
    print(ko_board)
    print(f"Ko point: {ko_board.ko_point} (should be None)")
    print(f"Is B playing B3 (2,1) legal now? {ko_board.is_legal_move(ko_board.coords_to_idx(2,1))}") # Should be True


    # Add a coords_to_idx helper to GoBoard for easier testing
    def coords_to_idx(self, r, c):
        return r * self.size + c
    GoBoard.coords_to_idx = coords_to_idx

    # Test Suicide
    # . . . . .
    # . X . . . (1,1) B
    # X O X . . (2,0) B, (2,1) W, (2,2) B
    # . X . . . (3,1) B
    # . . . . .
    # W tries to play at (2,1) - this is suicide
    suicide_board = GoBoard(5)
    suicide_board.play_move(suicide_board.coords_to_idx(1,1)) # B
    suicide_board.play_move(suicide_board.coords_to_idx(0,0)) # W dummy
    suicide_board.play_move(suicide_board.coords_to_idx(2,0)) # B
    suicide_board.play_move(suicide_board.coords_to_idx(0,1)) # W dummy
    suicide_board.play_move(suicide_board.coords_to_idx(2,2)) # B
    suicide_board.play_move(suicide_board.coords_to_idx(0,2)) # W dummy
    suicide_board.play_move(suicide_board.coords_to_idx(3,1)) # B
    print("Suicide Test Setup:")
    print(suicide_board) # W to play

    # W tries to play at (2,1)
    suicide_move_idx = suicide_board.coords_to_idx(2,1)
    print(f"Is W playing at (2,1) (idx {suicide_move_idx}) legal? {suicide_board.is_legal_move(suicide_move_idx)}") # Should be False

    # Test suicide that captures (not suicide)
    # . O . . . (0,1) W
    # O X O . . (1,0) W, (1,1) B, (1,2) W
    # . O . . . (2,1) W
    # . . . . .
    # B plays at (1,1) - this would be suicide, but it captures the surrounding Os
    cap_suicide_board = GoBoard(5)
    cap_suicide_board.to_play = -1 # W starts
    cap_suicide_board.play_move(cap_suicide_board.coords_to_idx(0,1)) #W
    cap_suicide_board.play_move(cap_suicide_board.coords_to_idx(0,0)) #B dummy
    cap_suicide_board.play_move(cap_suicide_board.coords_to_idx(1,0)) #W
    cap_suicide_board.play_move(cap_suicide_board.coords_to_idx(0,2)) #B dummy
    cap_suicide_board.play_move(cap_suicide_board.coords_to_idx(1,2)) #W
    cap_suicide_board.play_move(cap_suicide_board.coords_to_idx(0,3)) #B dummy
    cap_suicide_board.play_move(cap_suicide_board.coords_to_idx(2,1)) #W
    print("Capture-Suicide Test Setup:")
    print(cap_suicide_board) # B to play

    # B plays at (1,1)
    cap_suicide_move_idx = cap_suicide_board.coords_to_idx(1,1)
    print(f"Is B playing at (1,1) (idx {cap_suicide_move_idx}) legal? {cap_suicide_board.is_legal_move(cap_suicide_move_idx)}") # Should be True
    if cap_suicide_board.is_legal_move(cap_suicide_move_idx):
        cap_suicide_board.play_move(cap_suicide_move_idx)
        print("Board after B plays at (1,1):")
        print(cap_suicide_board)
        print(f"Captured by Black: {cap_suicide_board.captured_stones[1]}")

```
