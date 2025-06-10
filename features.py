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
    # For simplicity in this first pass, we will call count_liberties per stone.
    for r in range(size):
        for c in range(size):
            stone_color = board.board[r, c].item()
            
            if stone_color == current_player_color:
                libs = board.count_liberties(r, c)
                if libs > 0:
                    lib_plane_idx = 2 + min(libs, 8)
                    if model_channels > lib_plane_idx:
                        features[0, lib_plane_idx, r, c] = 1.0
            
            elif stone_color == opponent_color:
                libs = board.count_liberties(r, c)
                if libs > 0:
                    lib_plane_idx = 10 + min(libs, 8)
                    if model_channels > lib_plane_idx:
                        features[0, lib_plane_idx, r, c] = 1.0

    # --- Ko Point (Plane 19) ---
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
    player_to_play_plane_idx = 21
    if model_channels > player_to_play_plane_idx:
        features[0, player_to_play_plane_idx, :, :] = float(board.to_play)

    # All other planes will remain zero if not explicitly filled.
    return features