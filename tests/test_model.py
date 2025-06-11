import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from model import KataGoModel, ModelConfig  # noqa: E402
from board import GoBoard  # noqa: E402
from features import extract_katago_features  # noqa: E402


def test_forward_shapes_and_masking():
    config = ModelConfig(board_size=5, in_channels=22)
    model = KataGoModel(config)
    board = GoBoard(size=config.board_size)
    features = extract_katago_features(board, config.in_channels)
    mask = torch.ones(1, config.board_size, config.board_size)
    mask[0, 0, 0] = 0  # illegal move at (0,0)
    outputs = model(features, board_mask=mask)
    policy = outputs["policy_logits"]
    assert policy.shape == (1, config.board_size * config.board_size + 1)
    # masked move should produce a very low logit (effectively -inf)
    assert policy[0, 0].item() < -10


def test_training_step_updates_weights():
    config = ModelConfig(board_size=5, in_channels=22)
    model = KataGoModel(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    board = GoBoard(size=config.board_size)
    features = extract_katago_features(board, config.in_channels)
    mask = torch.ones(1, config.board_size, config.board_size)
    outputs = model(features, board_mask=mask)
    target_move = torch.tensor([board.PASS_MOVE])
    policy_loss = F.nll_loss(outputs["policy_logits"], target_move)
    value_loss = F.mse_loss(outputs["score_mean"].squeeze(), torch.tensor(0.0))
    loss = policy_loss + value_loss
    loss.backward()
    before = [p.clone() for p in model.parameters()]
    optimizer.step()
    after = list(model.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed
