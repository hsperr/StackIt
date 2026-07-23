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


def play_game(evaluator, cfg, rng):
    """Play one self-play game. Returns (examples, record) where
    examples = list of (planes, pi, player) and record is a light dict for the
    dashboard (moves + board size + winner)."""
    mcts = MCTS(evaluator, cfg)
    board = Board(cfg.board_size, cfg.board_size)
    positions = []            # (planes, pi, player)
    moves = []

    for ply in range(cfg.max_game_plies):
        if terminal_value(board) is not None:
            break
        counts, _ = mcts.search(board, add_noise=True)
        total = counts.sum()
        if total == 0:                        # no legal moves (shouldn't happen here)
            break
        pi = counts / total

        positions.append((encode(board), pi.astype(np.float32), board.current_player))

        if ply < cfg.temp_moves:              # tau=1: sample proportional to visits
            action = int(rng.choice(len(counts), p=pi))
        else:                                 # tau->0: greedy
            action = int(np.argmax(counts))
        x, y = index_to_move(action, cfg.board_size)
        moves.append((x, y))
        board.move(x, y)

    winner = game_winner(board)
    examples = []
    for planes, pi, player in positions:
        if winner == 0:
            z = 0.0
        else:
            z = 1.0 if player == winner else -1.0
        examples.append((planes, pi, z))

    record = {"moves": moves, "size": cfg.board_size, "winner": winner,
              "plies": len(moves)}
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
