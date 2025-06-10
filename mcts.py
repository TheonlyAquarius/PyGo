from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import torch

from model import KataGoModel, ModelConfig
from board import GoBoard # Use the new GoBoard
from features import extract_katago_features # Use the new feature extraction


@dataclass
class MCTSNode:
    """A node in the Monte Carlo Tree Search.

    Attributes
    ----------
    board : GoBoard # Changed from MinimalGoBoard
        Board position represented by this node.
    parent : Optional[MCTSNode]
        Link to parent node or ``None`` if root.
    children : Dict[int, MCTSNode] = field(default_factory=dict)
        Mapping of action index to child nodes.
    N : int = 0
        Visit count.
    W : float = 0.0
        Total value accumulated through simulations.
    P : float = 1.0 # Prior probability for the action *leading* to this node's state
        Prior probability from the policy network for the move that led to this state.
    """

    board: GoBoard
    parent: Optional["MCTSNode"] = None
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)
    N: int = 0
    W: float = 0.0
    P: float = 1.0


class MCTS:
    """Simplified UCT Monte Carlo Tree Search using :class:`KataGoModel` and :class:`GoBoard`.

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
        if not node.children: # Should not happen if called correctly
            raise ValueError("Cannot select from a node with no children.")

        # actions = list(node.children.keys()) # Redundant, children values are MCTSNode instances
        children_nodes = list(node.children.values())

        # Calculate N, W, P for each child
        # N_parent is node.N (visit count of current node)
        # N_child is child.N (visit count of child node)
        # W_child is child.W (total value from child's perspective)
        # P_child is child.P (prior probability of selecting this child from parent)

        N_parent_sqrt = math.sqrt(max(node.N, 1)) # max(node.N, 1) to prevent div by zero if node.N is 0 (e.g. root before playouts)

        best_score = -float('inf')
        best_child = None

        for child_node in children_nodes:
            Q = child_node.W / child_node.N if child_node.N > 0 else 0.0
            # The value W in a child node is from its perspective.
            # If the parent is player A, child is player B. MCTS aims to maximize value for the current player at the node.
            # So, Q for player A from child (state B) should be - (value for B).
            # However, W is stored as "sum of outcomes from this node's perspective".
            # When backing up, v = -v. So child.W is already from parent's perspective if v was correctly flipped.
            # Let's assume W is stored such that higher is better for the player whose turn it was at the *child* node.
            # During backup: n.W += v (current node); v = -v (for parent).
            # So, if node is current player, and child is opponent's turn, child.W is sum of (-v_parent).
            # Q_child = child.W / child.N. This is the value from the child's perspective.
            # We need value from parent's perspective. So, Q_parent = -Q_child.
            # Let's re-evaluate the backup logic:
            # Backup: path is [root, n1, n2, ..., leaf]. value is from leaf's perspective.
            # for n in reversed(path): n.N +=1; n.W += v; v = -v.
            # So, leaf.W += v_leaf. parent_of_leaf.W += (-v_leaf).
            # This means W at each node is for the player *who made the move to get to that node*. This is not standard.
            # Standard: W is sum of values for player whose turn it is *at that node*.
            # If W is for player whose turn it is at node:
            #   Q = node.W / node.N (value for current player)
            #   Then for parent considering child: Q_parent_perspective = - (child.W / child.N)
            # Let's adjust the selection to reflect this standard interpretation.
            # The P attribute of a child node is the prior for the *action* that led to it.

            # Simpler: Assume W is always from the perspective of the player whose turn it is at the node.
            # When selecting a child of 'node', 'node' is current player. Child represents opponent's turn.
            # So we want to pick child that MINIMIZES child.W / child.N (or maximizes -(child.W/child.N) )
            # This seems to be a common source of confusion. AlphaZero PUCT: Q(s,a) + U(s,a)
            # Q(s,a) is the mean action value of taking action a from state s.
            # N(s,a) is visit count of action a from state s.
            # W(s,a) is total action value of action a from state s. Q(s,a) = W(s,a)/N(s,a)
            # P(s,a) is prior probability of action a from policy network.
            # U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)) where N(s) = sum over b of N(s,b)
            # In this code, node.N is N(s). child.N is N(s,a). child.P is P(s,a).
            # child.W seems to be W(s,a), but needs to be from perspective of player at s.

            # Let's assume child.W is the sum of values from the perspective of the player at child.board.to_play.
            # Since child.board.to_play is the opponent of node.board.to_play,
            # the value for player at 'node' is - (child.W / child.N)
            Q_for_parent = (-child_node.W / child_node.N) if child_node.N > 0 else 0.0

            U = self.c_puct * child_node.P * N_parent_sqrt / (1 + child_node.N)
            score = Q_for_parent + U

            if score > best_score:
                best_score = score
                best_child = child_node

        if best_child is None: # Should only happen if children list was empty, but we checked. Or if all scores are -inf.
             # Fallback if all children have N=0 (e.g. first visit to this node's children)
             # then Q=0 for all, pick based on U. If P is also 0 for some, issues.
             # For now, let's assume P is non-zero for legal moves.
             if not children_nodes: # Should be caught earlier
                 raise Exception("No children to select from, this should not happen here.")
             # This case might occur if all Q_for_parent are -inf and all U are 0 or -inf.
             # This implies an issue with P (priors) or N_parent_sqrt.
             # A simple fallback if all scores are terrible (e.g. all children lead to loss):
             # just pick one, e.g. the first one or one with highest P.
             # For now, rely on argmax behavior with floats.
             # If all scores are identical (e.g. all 0 at start), torch.argmax picks first.
             # This should be okay.
             # If best_child is still None, this indicates an issue in logic or extreme values.
             # For robustness, if best_child is None after loop, pick first child.
             # This path should ideally not be taken if P values are reasonable.
             # This situation could also arise if all children nodes have N=0, then Q=0.
             # Then, selection is based on U = c_puct * P * sqrt(N_parent) / 1.
             # So, child with highest P is chosen. This is correct.
             # The code `torch.argmax(Q + U).item()` from original code would handle this fine.
             # My loop is equivalent.
             if not children_nodes: return children_nodes[0] # Should not be reachable

        return best_child


    def search(self, board: GoBoard) -> int: # board is GoBoard, no mask
        """Run MCTS playouts from ``board`` and return the best move index."""

        root = MCTSNode(board.copy()) # Use GoBoard.copy()

        for _ in range(self.n_playouts):
            node = root
            path = [node]

            # Selection
            while node.children: # While node is not a leaf in the *current search tree*
                if node.board.is_terminal(): # If actual game terminal state, break selection
                    break
                node = self._uct_select(node)
                path.append(node)

            # Expansion & Evaluation
            # If node is terminal in game logic, its value is determined by game rules (win/loss/draw)
            # For Go, this is complex (scoring). For now, model's value head will estimate.
            # If we expanded down to a game terminal state, the model's value output will be used.

            # features = extract_katago_features(node.board, self.config.in_channels) # Use new extraction
            # features = features.to(self.device)

            # Create legal moves mask for the model's policy head
            # Shape (1, board_size, board_size), 1.0 for legal, 0.0 for illegal
            # The model's policy head might expect -inf for illegal logits.
            # For now, let's create a binary mask and assume model handles it or we adjust model output later.
            # The model in `model.py` applies this mask by adding it to logits (expects 0 for legal, -inf for illegal).

            policy_output_size = self.config.board_size * self.config.board_size + 1 # board_size^2 + 1 for pass

            # This mask is for the *output* of the policy head (logits)
            # It should be (1, policy_output_size)
            # 0 for allowed moves, -infinity for disallowed moves.
            policy_mask = torch.full((1, policy_output_size), 0.0, device=self.device, dtype=torch.float32)
            for move_idx in range(policy_output_size):
                 if not node.board.is_legal_move(move_idx):
                    policy_mask[0, move_idx] = float('-inf')

            # Prepare input features for the model
            current_features = extract_katago_features(node.board, self.config.in_channels)
            current_features = current_features.to(self.device)

            with torch.no_grad():
                # Pass policy_mask to the model if it supports it directly.
                # The KataGoModel in model.py takes board_mask (spatial) not policy_mask (flat).
                # It constructs its own policy_mask from board_mask internally before softmax.
                # So we need to provide a spatial board_mask (0 for illegal, 1 for legal).
                spatial_board_mask = torch.zeros(1, node.board.size, node.board.size, device=self.device, dtype=torch.float32)
                for r_idx in range(node.board.size):
                    for c_idx in range(node.board.size):
                        move_idx = r_idx * node.board.size + c_idx
                        if node.board.is_legal_move(move_idx):
                            spatial_board_mask[0, r_idx, c_idx] = 1.0
                # Pass move is implicitly handled by model if board_mask only covers board points.
                # The model.py KataGoModel expects board_mask for policy head.

                outputs = self.model(current_features, board_mask=spatial_board_mask, use_inference_heads=True)

            policy_logits = outputs["policy_logits"].squeeze(0) # Shape: (policy_output_size,)
            value = outputs["score_mean"].squeeze().item() # Scalar value

            # Apply policy_mask to logits before softmax, if not done by model
            # model.py already does this using the spatial_board_mask
            # priors = torch.softmax(policy_logits, dim=0) # policy_logits should already be masked
            priors = torch.softmax(policy_logits, dim=-1)


            if not node.board.is_terminal():
                # Expand children for all legal moves
                for move in node.board.legal_moves(): # Uses new GoBoard.legal_moves()
                    if move not in node.children:
                        next_board_state = node.board.copy()
                        next_board_state.play_move(move)
                        # Prior P for the child is P(action_that_leads_to_child | parent_state)
                        node.children[move] = MCTSNode(next_board_state, parent=node, P=priors[move].item())

            # Backup: Propagate value (-v for parent) up the path
            # v is from the perspective of the player *at the current node (node.board.to_play)*
            v = value
            for selected_node_in_path in reversed(path):
                selected_node_in_path.N += 1
                # W should accumulate value from perspective of player at selected_node_in_path
                selected_node_in_path.W += v
                v = -v # Flip value for the parent

        if not root.children:
            # This can happen if n_playouts is 0 or if the root is a terminal state
            # and no children were ever expanded (e.g. if expansion only happens for non-terminal)
            # If root is terminal, it should ideally have a value, but no best move.
            # GTP spec requires a move or 'resign'. If no moves, pass.
            if board.is_terminal(): # board is initial board passed to search
                 return board.PASS_MOVE # Or handle terminal state appropriately

            # If no children after playouts (e.g. n_playouts=0 or immediate terminal state)
            # Fallback: try to list legal moves from the root board.
            # This part of code means "what move to play *from root*".
            # If no children were generated, it means we didn't even run the loop once for the root.
            # This implies n_playouts was < 1 or some other early exit.
            # A robust MCTS should at least evaluate root, get priors, and pick based on that if n_playouts is very small.
            # The loop for _ in range(self.n_playouts) handles this.
            # So, if root.children is empty, it implies root state itself is terminal and expansion didn't add children.
            # Or, if all legal moves from root lead to immediate terminal states.
            # This situation needs graceful handling. For now, if no children, pass.
            return board.PASS_MOVE


        # Select best move from root based on visit counts (most robust)
        # or max value (Q value), or a combination. Typically visit counts.
        best_move = -1
        max_visits = -1
        for move, child_node in root.children.items():
            if child_node.N > max_visits:
                max_visits = child_node.N
                best_move = move

        if best_move == -1: # Should not happen if root.children is not empty
            # Fallback if all children have 0 visits (e.g. if n_playouts was too small to visit children of root's children)
            # In such a case, priors from the first evaluation of root would be in root.children[move].P
            # We could pick based on highest prior if all N are 0.
            # However, the MCTS loop ensures each child of root is created with a P.
            # If n_playouts = 1, one path is explored. Root's children are created. One of them gets N=1.
            # So this fallback should ideally not be needed.
            # If there are children, at least one must have N>0 if any playouts happened.
            # If n_playouts = 0, this function should probably not be called or handle it differently.
            # If it's still -1, it means root.children was not empty but all N were <= -1 (impossible).
            # Or, if root.children was empty, which is handled above.
            # One last check: if all children have N=0 (e.g. after one simulation that ends at root)
            # then pick child with highest prior P.
            if max_visits == 0:
                max_prior = -1.0
                for move, child_node in root.children.items():
                    if child_node.P > max_prior:
                        max_prior = child_node.P
                        best_move = move
            if best_move == -1 and root.children: # Still no best move, pick first available
                best_move = list(root.children.keys())[0]
            elif not root.children: # no children at all
                return board.PASS_MOVE


        return best_move


if __name__ == "__main__":
    # Ensure ModelConfig and KataGoModel are imported correctly
    # from model import ModelConfig, KataGoModel
    # from board import GoBoard

    print("MCTS Self-Test with GoBoard")
    config = ModelConfig(board_size=5, in_channels=22) # features.py expects up to 22
    model = KataGoModel(config) # model.py uses config.in_channels

    # Check if config.in_channels is what extract_katago_features expects
    # The features.py uses model_channels argument.
    # ModelConfig has in_channels. Let's assume they are the same.

    board = GoBoard(size=5)

    # Play a few moves on the board for a more interesting test
    board.play_move(board.coords_to_idx(2,2)) # B
    board.play_move(board.coords_to_idx(2,3)) # W
    board.play_move(board.coords_to_idx(3,2)) # B
    board.play_move(board.coords_to_idx(3,3)) # W
    print("Board state for MCTS test:")
    print(board)

    mcts = MCTS(model, config, n_playouts=16) # Reduced playouts for faster test

    print("Running MCTS search...")
    # Pass the GoBoard instance directly, no mask tensor
    best_move_idx = mcts.search(board)

    if best_move_idx == board.PASS_MOVE:
        print(f"Best move: PASS")
    else:
        r, c = best_move_idx // board.size, best_move_idx % board.size
        print(f"Best move index: {best_move_idx} ({r},{c})")

    # Example of playing the move
    # board.play_move(best_move_idx)
    # print("\nBoard after MCTS move:")
    # print(board)

