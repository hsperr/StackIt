"""What is the highest top-1 ANY imitator could score against this teacher?

The policy target is one-hot on AlphaBeta's chosen move. But AlphaBeta is budgeted
on wall-clock and breaks ties by move order, so ask it the same question twice and
it does not always answer the same way. Every such disagreement is a label a
perfect student still gets 'wrong'. AlphaBeta's agreement with ITSELF is therefore
the ceiling on top-1, and chasing 0.9 is only sensible if that ceiling is above 0.9.

Also reports agreement between 0.05s (the teacher we trained on) and 0.3s, i.e.
how much of the label is budget-specific rather than about the position.
"""
import sys, json
sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')
import numpy as np
from board import Board
from alphazero.arena_eval import CAlphaBetaPlayer

BIN = '/Users/hsperr/code/youtube_coding/StackIt/c_engine/stackit_new'
N = 5
n_pos = int(sys.argv[1]) if len(sys.argv) > 1 else 400

a1 = CAlphaBetaPlayer(0.05, binary=BIN)
a2 = CAlphaBetaPlayer(0.05, binary=BIN)
a2.binary = BIN + "#2"          # force a SECOND process, not the shared one
a2.binary = BIN                 # (shared dict is keyed by path; use a fresh object)
slow = CAlphaBetaPlayer(0.3, binary=BIN)

rng = np.random.default_rng(11)
same_fast = same_slow = tot = 0
for k in range(n_pos):
    b = Board(N, N)
    for _ in range(int(rng.integers(2, 30))):        # random position, any depth
        legal = b.possible_moves()
        if not legal or b.winning_player():
            break
        b.move(*legal[rng.integers(len(legal))])
    if b.winning_player() or not b.possible_moves():
        continue
    m1 = a1.move(b, rng)
    m2 = a2.move(b, rng)
    m3 = slow.move(b, rng)
    if m1 is None or m2 is None or m3 is None:
        continue
    tot += 1
    same_fast += (tuple(m1) == tuple(m2))
    same_slow += (tuple(m1) == tuple(m3))
print(f"{tot} positions")
print(f"  AlphaBeta 0.05s vs itself 0.05s : {same_fast/tot:.3f}   <- ceiling on top-1")
print(f"  AlphaBeta 0.05s vs      0.3s    : {same_slow/tot:.3f}")
