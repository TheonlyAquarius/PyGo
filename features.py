import torch
from board import GoBoard # Assuming board.py is in the same directory or accessible
from typing import Set, Tuple, List, Dict, Optional

def extract_katago_features(board: GoBoard, model_channels: int) -> torch.Tensor:
    """
    Extracts spatial features from a GoBoard state, similar to KataGo's input representation.

    Args:
        board (GoBoard): The current Go board state.
        model_channels (int): The number of expected input channels for the KataGo model.

    Returns:
        torch.Tensor: A tensor of shape (1, model_channels, board.size, board.size).
    """
    if not isinstance(board, GoBoard):
        raise ValueError("Input 'board' must be an instance of GoBoard.")
    if not isinstance(model_channels, int) or model_channels <= 0:
        raise ValueError("model_channels must be a positive integer.")

    size = board.size
    features = torch.zeros((1, model_channels, size, size), dtype=torch.float32)

    current_player_color = board.to_play
    opponent_color = -board.to_play

    # --- Base Features (Planes 0, 1, 2) ---
    # Plane 0: Player Stones
    if model_channels > 0:
        features[0, 0, :, :] = (board.board == current_player_color).float()
    
    # Plane 1: Opponent Stones
    if model_channels > 1:
        features[0, 1, :, :] = (board.board == opponent_color).float()

    # Plane 2: Empty Points
    if model_channels > 2:
        features[0, 2, :, :] = (board.board == 0).float()

    # --- Liberty-based Features (Planes 3-10 for player, 11-18 for opponent) ---
    # Max 8 planes for liberties (1 to 7, and 8+)
    # Player liberties: channels 3 to 10
    # Opponent liberties: channels 11 to 18
    
    # Iterate through each point on the board once to calculate liberties
    # This is more efficient than calling board.count_liberties() for each stone repeatedly
    # However, board.count_liberties itself uses _get_group, which finds all stones in a group.
    # To avoid redundant _get_group calls for stones in the same group, we can cache group liberties.
    
    # For simplicity in this first pass, we will call count_liberties per stone.
    # Optimization can be added later if performance is an issue.
    
    for r in range(size):
        for c in range(size):
            stone_color = board.board[r, c].item()
            
            if stone_color == current_player_color:
                libs = board.count_liberties(r, c)
                # Player liberty planes start at index 3
                # min(libs, 8) -> 1 to 8. To map to planes 3 to 10: (2 + min(libs,8))
                # libs = 1 -> plane 3 (2+1)
                # libs = 8 -> plane 10 (2+8)
                if libs > 0: # only mark if there are liberties (stone exists)
                    lib_plane_idx = 2 + min(libs, 8)
                    if model_channels > lib_plane_idx:
                        features[0, lib_plane_idx, r, c] = 1.0
            
            elif stone_color == opponent_color:
                libs = board.count_liberties(r, c)
                # Opponent liberty planes start at index 11 (3 + 8)
                # min(libs, 8) -> 1 to 8. To map to planes 11 to 18: (10 + min(libs,8))
                # libs = 1 -> plane 11 (10+1)
                # libs = 8 -> plane 18 (10+8)
                if libs > 0: # only mark if there are liberties (stone exists)
                    lib_plane_idx = 10 + min(libs, 8)
                    if model_channels > lib_plane_idx:
                        features[0, lib_plane_idx, r, c] = 1.0

    # --- Ko Point (Plane 19) ---
    # Current channel count used: 3 (base) + 8 (player_lib) + 8 (opp_lib) = 19
    ko_plane_idx = 19
    if model_channels > ko_plane_idx and board.ko_point is not None:
        ko_r, ko_c = board.ko_point // size, board.ko_point % size
        features[0, ko_plane_idx, ko_r, ko_c] = 1.0

    # --- Last Move Location (Plane 20) ---
    last_move_plane_idx = 20
    if model_channels > last_move_plane_idx and board.history:
        last_player, last_move_idx = board.history[-1]
        if last_move_idx != board.PASS_MOVE:
            last_r, last_c = last_move_idx // size, last_move_idx % size
            features[0, last_move_plane_idx, last_r, last_c] = 1.0
            
    # --- Player to Play Plane (Plane 21) ---
    # Indicates whose turn it is. Filled with 1.0 if current player is Black, -1.0 if White.
    # Or 1.0 for current player, 0.0 for other schemes. KataGo typically uses 0s and 1s.
    # Let's use a single plane: 1.0 if Black to play, 0.0 if White to play (as per some interpretations)
    # Or, more directly, fill plane with board.to_play (1.0 for black, -1.0 for white)
    player_to_play_plane_idx = 21
    if model_channels > player_to_play_plane_idx:
        # features[0, player_to_play_plane_idx, :, :] = float(board.to_play)
        # KataGo often uses binary indicators. Let's use two planes if budget allows or one if not.
        # For now, let's use a single plane indicating the current player's color (1.0 or -1.0).
        # This matches the "sense of color" features in some models.
        features[0, player_to_play_plane_idx, :, :] = float(board.to_play)


    # --- Additional History Planes (Example: up to last 5 moves) ---
    # Example: Plane 22 for 2nd to last move, Plane 23 for 3rd, etc.
    # Requires model_channels > 22, > 23, etc.
    # Start history planes from channel `base_history_idx`. Last move is already at `last_move_plane_idx`.
    # Let's refine this for multiple history moves if channels allow.
    # If last_move_plane_idx is 20.
    # history_planes_start_idx = 20 (for current player's last move)
    # history_planes_opponent_start_idx = history_planes_start_idx + num_history_planes_per_player
    
    # Simplified: Last N moves, regardless of player.
    # Plane 20 = last move (already done)
    # Plane 22 = 2nd to last move (if model_channels > 22)
    # Plane 23 = 3rd to last move (if model_channels > 23)
    # ...
    # This is a simple linear history. KataGo's history features are more complex (e.g. N recent moves by current player, N recent moves by opponent)
    # For now, let's stick to the single "last move" at plane 20.
    # If more history planes were to be added, the logic would be:
    # current_history_plane = last_move_plane_idx # 20
    # for i, (player, move_idx) in enumerate(reversed(board.history)):
    #     if i == 0: # Already handled by last_move_plane_idx
    #         continue
    #     target_plane = some_base_index_for_further_history + i -1 
    #     if model_channels > target_plane and move_idx != board.PASS_MOVE:
    #          move_r, move_c = move_idx // size, move_idx % size
    #          features[0, target_plane, move_r, move_c] = 1.0
    #     if target_plane >= model_channels -1 : break # stop if no more channels

    # All other planes will remain zero if not explicitly filled.
    return features

if __name__ == '__main__':
    # Basic test
    from board import GoBoard
    
    test_board = GoBoard(19)
    # Play some moves
    # B: D4 (3,3) -> (19-4)*19+3 = 15*19+3 = 285+3 = 288
    # W: Q16 (15,15) -> (19-16)*19+15 = 3*19+15 = 57+15 = 72
    # B: Pass
    # W: K10 (9,9) -> (19-10)*19+9 = 9*19+9 = 171+9 = 180 (ko point for example)
    
    print("Creating board and playing some moves...")
    # Helper for tests
    def c2i(r_chess, c_char, s=test_board.size):
        # r_chess is 1-indexed from bottom, c_char is 'A'-'T' (skipping 'I')
        col_idx = ord(c_char.upper()) - ord('A')
        if col_idx >= 8: # 'I' is skipped
            col_idx -=1
        row_idx = s - r_chess # 0-indexed from top
        return row_idx * s + col_idx

    # test_board.play_move(c2i(4, 'D')) # B at D4 (r=15,c=3 in 0-indexed from top)
    # test_board.play_move(c2i(16, 'Q'))# W at Q16 (r=3,c=15)
    # test_board.play_move(test_board.PASS_MOVE) # B Pass
    # test_board.ko_point = c2i(10,'K') # W K10 (r=9,c=9) - set manually for testing ko plane
    # test_board.play_move(c2i(10,'K')) # W K10 - this would be illegal if ko rule applied before this play

    # Simpler setup for testing specific features
    b = GoBoard(5)
    # (0,0) B, (0,1) W, (1,0) B with 1 liberty
    # (2,2) B, (2,3) W, (3,2) B, (3,3) W - W at (3,3) has 2 libs
    # (4,4) B - pass - W to play, (4,4) is last move
    # Ko at (0,2)
    
    b.play_move(b.coords_to_idx(0,0)) # B
    b.play_move(b.coords_to_idx(0,1)) # W
    b.play_move(b.coords_to_idx(1,0)) # B - stone (0,0) has 2 libs, (1,0) has 2 libs
                                     # W stone (0,1) has 2 libs
    
    # Check stone (0,0) for B (current player is W, so B is opponent)
    # (0,0) is opponent, libs = 2. Plane 10 + 2 = 12
    
    b.play_move(b.coords_to_idx(2,2)) # W
    b.play_move(b.coords_to_idx(2,3)) # B
    b.play_move(b.coords_to_idx(3,2)) # W
    b.play_move(b.coords_to_idx(3,3)) # B - stone (3,3) has 2 libs. Current player W. (3,3) is opponent. Plane 10+2=12
                                     # stone (2,3) B has 3 libs.
                                     # stone (2,2) W has 3 libs. Plane 2+3 = 5
                                     # stone (3,2) W has 3 libs. Plane 2+3 = 5
    
    # B plays pass
    b.play_move(b.PASS_MOVE) # B plays pass, W to play. Last move was pass.
    
    # W plays (4,4)
    b.play_move(b.coords_to_idx(4,4)) # W plays (4,4). B to play. Last move (4,4) by W.
    
    b.ko_point = b.coords_to_idx(0,2) # Manually set ko point for testing

    print("\nBoard state:")
    print(b) # B to play

    print(f"\nTesting with model_channels = 22")
    features_22 = extract_katago_features(b, 22)
    print(f"Output shape: {features_22.shape}")

    # Player is Black (1)
    # Opponent is White (-1)

    # Plane 0: Player Stones (Black)
    print(f"\nPlayer (Black) stones (Plane 0):")
    print(features_22[0,0,:,:])
    # Expected: (0,0)B, (1,0)B, (2,3)B, (3,3)B

    # Plane 1: Opponent Stones (White)
    print(f"\nOpponent (White) stones (Plane 1):")
    print(features_22[0,1,:,:])
    # Expected: (0,1)W, (2,2)W, (3,2)W, (4,4)W
    
    # Plane 2: Empty
    print(f"\nEmpty points (Plane 2):")
    print(features_22[0,2,:,:])

    # Player (Black) stone liberties
    # (0,0)B has 2 libs (adj to (0,1)W, (1,0)B). Libs at (1,1), (0,2) if empty...
    # Let's re-verify with current board:
    # B . O . .  (0,0)B, (0,1)W. (0,0)B has libs at (1,0)-B and empty (0,2) if ko is not there.
    # X . . . .  (1,0)B.
    # . . W B .  (2,2)W, (2,3)B
    # . . W B .  (3,2)W, (3,3)B
    # . . . . O  (4,4)W
    # B to play. Ko at (0,2)
    # Player (B) stones:
    # (0,0)B: libs at (1,1) (empty), (0,2) (empty, but ko). So 1 actual liberty?
    #   _get_group for (0,0)B: stones={(0,0),(1,0)}, libs={(1,1),(0,2),(2,0)}. Count = 3. Plane 2+3 = 5.
    # (1,0)B: same group. Plane 5.
    # (2,3)B: libs at (1,3),(2,4),(3,3)-B,(1,2). Count = 3. Plane 5.
    # (3,3)B: same group. Plane 5.
    print(f"\nPlayer (Black) stone with 3 libs (Plane 5 = 2+3):")
    print(features_22[0, 5, :, :])


    # Opponent (White) stone liberties
    # (0,1)W: libs at (0,0)-B,(1,1),(0,2)-ko. Count = 2. Plane 10+2 = 12.
    # (2,2)W: libs at (1,2),(2,1),(3,2)-W,(1,3). Count = 3. Plane 10+3 = 13.
    # (3,2)W: same group. Plane 13.
    # (4,4)W: libs at (3,4),(4,3). Count = 2. Plane 12.
    print(f"\nOpponent (White) stone with 2 libs (Plane 12 = 10+2):")
    print(features_22[0, 12, :, :])
    print(f"\nOpponent (White) stone with 3 libs (Plane 13 = 10+3):")
    print(features_22[0, 13, :, :])


    # Plane 19: Ko point
    print(f"\nKo point (Plane 19) at (0,2):")
    print(features_22[0, 19, :, :]) # Expected: 1.0 at (0,2)
    
    # Plane 20: Last move location
    # Last actual move was (4,4) by W.
    print(f"\nLast move (Plane 20) at (4,4):")
    print(features_22[0, 20, :, :]) # Expected: 1.0 at (4,4)

    # Plane 21: Player to Play Plane
    # B to play, so value should be 1.0
    print(f"\nPlayer to Play (Plane 21), B to play (all 1.0):")
    print(features_22[0, 21, :, :])

    print(f"\nTesting with model_channels = 3")
    features_3 = extract_katago_features(b, 3)
    print(f"Output shape: {features_3.shape}")
    assert features_3[0,0,0,0].item() == 1.0 # Player stone B at (0,0)
    assert features_3[0,1,0,1].item() == 1.0 # Opponent stone W at (0,1)
    assert features_3[0,2,0,3].item() == 1.0 # Empty at (0,3)
    if model_channels > 3: # Check that other channels are zero
      if features_3.shape[1] > 3:
        assert torch.all(features_3[0,3,:,:] == 0)

    print("\nBasic tests completed.")
```
