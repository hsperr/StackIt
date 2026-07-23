import numpy as np
import pytest

from board import Board
from alphazero.config import Config
from alphazero.encoding import (encode, legal_mask, action_index, index_to_move,
                                augment, NUM_PLANES)
from alphazero.net import StackNet, Evaluator
from alphazero.mcts_az import MCTS, terminal_value
from alphazero.selfplay import game_winner


def test_encode_is_one_hot_per_cell():
    b = Board(4, 4)
    b.move(0, 0)
    planes = encode(b)
    assert planes.shape == (NUM_PLANES, 4, 4)
    # exactly one plane active per cell
    assert np.all(planes.sum(axis=0) == 1)


def test_encode_is_canonical_to_side_to_move():
    b = Board(4, 4)
    b.move(0, 0)                       # p1 plays -> p2 to move; that cell has 3 blocks
    planes = encode(b)
    # p1's cell (count 3) is "theirs" for the player to move -> plane 4 + (3-1) = 6
    assert planes[6, 0, 0] == 1
    assert planes[2, 0, 0] == 0        # not on the "mine" count-3 plane


def test_action_index_roundtrip_and_mask():
    b = Board(4, 4)
    m = legal_mask(b)
    for (x, y) in b.possible_moves():
        idx = action_index(x, y, 4)
        assert m[idx]
        assert index_to_move(idx, 4) == (x, y)


def test_augment_keeps_state_policy_aligned():
    b = Board(4, 4)
    b.move(1, 1)
    planes = encode(b)
    pi = np.zeros(16, dtype=np.float32)
    pi[action_index(2, 1, 4)] = 1.0
    for p2, pi2 in augment(planes, pi, 4, 4):
        assert p2.shape == (NUM_PLANES, 4, 4)
        assert np.all(p2.sum(axis=0) == 1)
        assert pytest.approx(pi2.sum(), abs=1e-6) == 1.0


def test_terminal_value_domination():
    dominated = Board.from_string('2' + '11' * 16)   # p2 to move, p1 owns all
    assert dominated.winning_player() == 1
    assert terminal_value(dominated) == -1.0          # loss for the side to move


def test_evaluator_masks_illegal_and_normalizes():
    b = Board(4, 4)
    b.move(1, 1)                     # (1,1) now p1's; still legal for p1 but not p2
    net = StackNet(4, channels=16, res_blocks=2)
    priors, v = Evaluator(net, "cpu").infer(b)
    mask = legal_mask(b)
    assert pytest.approx(priors.sum(), abs=1e-4) == 1.0
    assert np.all(priors[~mask] == 0)
    assert -1.0 <= v <= 1.0


def test_mcts_visit_counts_sum_to_sims_and_are_legal():
    cfg = Config(num_simulations=30, channels=16, res_blocks=2)
    net = StackNet(4, cfg.channels, cfg.res_blocks)
    mcts = MCTS(Evaluator(net, "cpu"), cfg)
    b = Board(4, 4)
    counts, root = mcts.search(b, add_noise=True)
    assert counts.sum() == cfg.num_simulations
    legal = set(root.legal.tolist())
    for i, c in enumerate(counts):
        if c > 0:
            assert i in legal


def test_game_winner_by_boxes():
    b = Board.from_string('1' + '31' + '00' + '00' + '00')  # 2x2, p1 has boxes
    assert game_winner(b) == 1
