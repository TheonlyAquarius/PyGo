from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from model import KataGoModel, ModelConfig


class MinimalGoBoard:
    """A tiny Go board representation for testing MCTS.

    Parameters
    ----------
    size : int
        Board width/height. Only square boards are supported.
    """

    def __init__(self, size: int = 19):
        self.size = size
        self.board = torch.zeros(size, size, dtype=torch.int8)
        self.to_play = 1  # 1 for black, -1 for white
        self.passes = 0

    def copy(self) -> "MinimalGoBoard":
        new = MinimalGoBoard(self.size)
        new.board = self.board.clone()
        new.to_play = self.to_play
        new.passes = self.passes
        return new

    def legal_moves(self):
        moves = [i for i in range(self.size * self.size) if self.board.view(-1)[i] == 0]
        moves.append(self.size * self.size)  # pass
        return moves

    def play(self, move: int) -> None:
        if move == self.size * self.size:
            self.passes += 1
        else:
            self.passes = 0
            x = move % self.size
            y = move // self.size
            self.board[y, x] = self.to_play
        self.to_play *= -1

    def is_terminal(self) -> bool:
        return self.passes >= 2 or (self.board != 0).all()


def extract_features(board: MinimalGoBoard) -> torch.Tensor:
    """Return a simple feature tensor for ``board``.

    The tensor has shape ``(1, 28, S, S)`` and encodes black/white stones
    and the player to move.
    """

    size = board.size
    feats = torch.zeros(1, 28, size, size, dtype=torch.float32)
    feats[0, 0] = (board.board == 1).float()
    feats[0, 1] = (board.board == -1).float()
    if board.to_play == 1:
        feats[0, 2].fill_(1.0)
    return feats


@dataclass
class MCTSNode:
    """A node in the Monte Carlo Tree Search.

    Attributes
    ----------
    board : MinimalGoBoard
        Board position represented by this node.
    parent : Optional[MCTSNode]
        Link to parent node or ``None`` if root.
    children : Dict[int, MCTSNode]
        Mapping of action index to child nodes.
    N : int
        Visit count.
    W : float
        Total value accumulated through simulations.
    P : float
        Prior probability from the policy network.
    """

    board: MinimalGoBoard
    parent: Optional["MCTSNode"] = None
    children: Dict[int, "MCTSNode"] = None
    N: int = 0
    W: float = 0.0
    P: float = 1.0

    def __post_init__(self):
        if self.children is None:
            self.children = {}
        self.features: Optional[torch.Tensor] = None


class MCTS:
    """Simplified UCT Monte Carlo Tree Search using :class:`KataGoModel`.

    Parameters
    ----------
    model : KataGoModel
        Neural network for policy and value evaluation.
    config : ModelConfig
        Configuration matching the neural network architecture.
    c_puct : float, optional
        Exploration constant in the PUCT formula.
    n_playouts : int, optional
        Number of playouts to run during :meth:`search`.
    device : str, optional
        Torch device for inference. Defaults to ``"cpu"``.
    """

    def __init__(self, model: KataGoModel, config: ModelConfig, c_puct: float = 1.5, n_playouts: int = 1600, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.c_puct = c_puct
        self.n_playouts = n_playouts
        self.device = device

    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        """Select a child with maximum UCT score."""

        actions = list(node.children.keys())
        children = list(node.children.values())
        N = torch.tensor([c.N for c in children], dtype=torch.float32)
        W = torch.tensor([c.W for c in children], dtype=torch.float32)
        P = torch.tensor([c.P for c in children], dtype=torch.float32)
        Q = torch.where(N > 0, W / N, torch.zeros_like(W))
        U = self.c_puct * P * math.sqrt(max(node.N, 1)) / (1 + N)
        idx = torch.argmax(Q + U).item()
        return children[idx]

    def search(self, board: MinimalGoBoard, mask: torch.Tensor) -> int:
        """Run MCTS playouts from ``board`` and return the best move index."""

        root = MCTSNode(board.copy())
        board_mask = mask.to(self.device)

        for _ in range(self.n_playouts):
            node = root
            path = [node]

            # Selection
            while node.children:
                node = self._uct_select(node)
                path.append(node)

            # Expansion
            if node.features is None:
                node.features = extract_features(node.board)
            features = node.features.to(self.device)
            # TODO: switch to GPU inference when available
            with torch.no_grad():
                outputs = self.model(features, board_mask=board_mask.unsqueeze(0), use_inference_heads=True)
            policy_logits = outputs["policy_logits"]
            value = outputs["score_mean"].squeeze(0)
            priors = torch.softmax(policy_logits, dim=1).squeeze(0)

            if not node.board.is_terminal():
                for move in node.board.legal_moves():
                    if move not in node.children:
                        next_board = node.board.copy()
                        next_board.play(move)
                        node.children[move] = MCTSNode(next_board, parent=node, P=priors[move].item())

            # Backup
            v = value.item()
            for n in reversed(path):
                n.N += 1
                n.W += v
                v = -v

        if not root.children:
            return board.size * board.size
        actions = list(root.children.keys())
        visits = [root.children[a].N for a in actions]
        best_idx = visits.index(max(visits))
        return actions[best_idx]


if __name__ == "__main__":
    config = ModelConfig(board_size=5, in_channels=28)
    model = KataGoModel(config)
    board = MinimalGoBoard(size=5)
    mask = torch.ones(config.board_size, config.board_size, dtype=torch.float32)
    mcts = MCTS(model, config, n_playouts=100)
    best = mcts.search(board, mask)
    print("Best move index:", best)

