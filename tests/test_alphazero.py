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
    own = np.zeros(16, dtype=np.int64)
    own[action_index(1, 1, 4)] = 1                 # p1's cell owned by the mover
    for p2, pi2, own2 in augment(planes, pi, own, 4, 4):
        assert p2.shape == (NUM_PLANES, 4, 4)
        assert np.all(p2.sum(axis=0) == 1)
        assert pytest.approx(pi2.sum(), abs=1e-6) == 1.0
        assert own2.shape == (16,)
        assert own2.sum() == 1                      # the single owned cell is preserved


def test_augment_symmetric_state_keeps_distinct_labels():
    # An empty board is symmetric under all 8 dihedral ops, but a one-hot policy
    # is not — each orientation is a distinct training target and must be kept.
    # (Keying dedup on the state alone would collapse these to one example.)
    b = Board(3, 3)
    planes = encode(b)
    pi = np.zeros(9, dtype=np.float32)
    pi[action_index(0, 0, 3)] = 1.0          # a corner cell
    own = np.zeros(9, dtype=np.int64)
    out = list(augment(planes, pi, own, 3, 3))
    assert len(out) > 1                       # not collapsed onto a single orientation
    keys = {pi2.tobytes() for _, pi2, _ in out}
    assert len(keys) == len(out)              # every emitted target is distinct


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


def test_mcts_policy_is_normalized_legal_and_selects_a_legal_action():
    cfg = Config(num_simulations=32, channels=16, res_blocks=2)
    net = StackNet(4, cfg.channels, cfg.res_blocks)
    mcts = MCTS(Evaluator(net, "cpu"), cfg)
    b = Board(4, 4)
    pi, root = mcts.search(b, add_noise=True)
    # Gumbel completed policy: a probability distribution over legal actions
    assert pytest.approx(pi.sum(), abs=1e-6) == 1.0
    legal = set(root.legal.tolist())
    for i, p in enumerate(pi):
        if p > 0:
            assert i in legal
    # the action to PLAY is a legal action, and is exposed separately from pi
    assert root.selected_action in legal


def test_mcts_sequential_halving_spends_exact_budget():
    # Sequential Halving must consume EXACTLY num_simulations root visits — no
    # more (over-budget), no fewer (under-budget). Try several budgets, incl.
    # non-power-of-two, that don't divide the schedule evenly.
    for sims in (16, 25, 30, 32, 40, 100, 128):
        cfg = Config(num_simulations=sims, channels=16, res_blocks=2)
        net = StackNet(4, cfg.channels, cfg.res_blocks)
        mcts = MCTS(Evaluator(net, "cpu"), cfg)
        _, root = mcts.search(Board(4, 4), add_noise=True)
        assert int(root.child_N.sum()) == sims, f"budget {sims}"


def test_game_winner_by_boxes():
    b = Board.from_string('1' + '31' + '00' + '00' + '00')  # 2x2, p1 has boxes
    assert game_winner(b) == 1
