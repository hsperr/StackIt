"""Generate self-play games with MCTS and turn them into training examples.

Each visited position stores (encoded_state, pi, player), where pi is the
MCTS visit-count distribution at temperature 1 (the "improved policy" target).
When the game ends, z (the outcome from that position's player's view) is
filled in. Moves are sampled with temperature 1 for the first `temp_moves`
plies (exploration/data diversity), then greedily.
"""
import numpy as np

from board import Board
from .encoding import encode, index_to_move, augment
from .mcts_az import MCTS, terminal_value


def game_winner(board):
    """0 draw / 1 / 2 — domination first, else box count (arena rules)."""
    w = board.winning_player()
    if w:
        return w
    b1, b2 = board.boxes_for(1), board.boxes_for(2)
    return 1 if b1 > b2 else 2 if b2 > b1 else 0


def play_game(evaluator, cfg, rng, opponent_ev=None):
    """Play one self-play game and return (examples, record).

    Exploration: the first `opening_random_plies` moves are uniformly random (and
    not recorded), so games start from diverse positions instead of collapsing
    onto one line.

    If `opponent_ev` is given, `evaluator` plays one (random) colour and the
    opponent net plays the other — and ONLY the main net's positions are recorded
    (on-policy targets for the net being trained), while the varied opponent
    still diversifies the trajectories."""
    n = cfg.board_size
    mcts = MCTS(evaluator, cfg)
    mcts_opp = MCTS(opponent_ev, cfg) if opponent_ev is not None else mcts
    main_color = 1 if opponent_ev is None else (1 if rng.random() < 0.5 else 2)

    board = Board(n, n)
    positions = []            # (planes, pi, player) — main net's positions only
    moves = []

    for ply in range(cfg.max_game_plies):
        if terminal_value(board) is not None:
            break
        mover = board.current_player

        if ply < cfg.opening_random_plies:            # random, unrecorded opening
            legal = board.possible_moves()
            x, y = legal[rng.integers(len(legal))]
            moves.append((x, y))
            board.move(x, y)
            continue

        is_main = opponent_ev is None or mover == main_color
        counts, _ = (mcts if is_main else mcts_opp).search(board, add_noise=True)
        total = counts.sum()
        if total == 0:
            break
        pi = counts / total

        if is_main:                                   # record only the main net
            positions.append((encode(board), pi.astype(np.float32), mover))

        if ply < cfg.temp_moves:                      # tau=1: sample by visits
            action = int(rng.choice(len(counts), p=pi))
        else:                                         # tau->0: greedy
            action = int(np.argmax(counts))
        x, y = index_to_move(action, n)
        moves.append((x, y))
        board.move(x, y)

    winner = game_winner(board)
    examples = []
    for planes, pi, player in positions:
        z = 0.0 if winner == 0 else (1.0 if player == winner else -1.0)
        examples.append((planes, pi, z))

    record = {"moves": moves, "size": n, "winner": winner, "plies": len(moves)}
    return examples, record


def expand_symmetries(examples, cfg):
    """Apply dihedral augmentation to a list of (planes, pi, z)."""
    if not cfg.augment_symmetries:
        return list(examples)
    out = []
    n = cfg.board_size
    for planes, pi, z in examples:
        for p2, pi2 in augment(planes, pi, n, n):
            out.append((p2, pi2.astype(np.float32), z))
    return out
