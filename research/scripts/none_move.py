"""Find the position where AZPlayer returns None on a board with legal moves.

The new abort guard in parallel.py fired instantly on a v45-vs-v3 match:
`player 2 returned None with 23 legal moves available`. AZPlayer returns None
only when the searched policy sums to zero, so this reproduces that in-process
and dumps everything about the offending root.
"""
import sys
import numpy as np

sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')

from board import Board
from alphazero.config import Config
from alphazero.metrics import load_net
from alphazero.net import Evaluator
from alphazero.arena_eval import AZPlayer
from alphazero.mcts_az import terminal_value

A = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints_cover/versions/v45.pt'
B = sys.argv[2] if len(sys.argv) > 2 else 'checkpoints_cover/versions/v3.pt'
GAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 20
SIMS = int(sys.argv[4]) if len(sys.argv) > 4 else 128

cfg = Config(channels=64, res_blocks=4, num_simulations=SIMS,
             eval_simulations=SIMS, match_random_plies=3)
na, _ = load_net(A, 'cpu')
nb, _ = load_net(B, 'cpu')
pa = AZPlayer(Evaluator(na, 'cpu'), cfg)
pb = AZPlayer(Evaluator(nb, 'cpu'), cfg)

hits = 0
for g in range(GAMES):
    rng = np.random.default_rng(1000 + g)
    p1, p2 = (pa, pb) if g % 2 == 0 else (pb, pa)
    p1.reset(); p2.reset()
    b = Board(cfg.board_size, cfg.board_size)
    for _ in range(cfg.match_random_plies):
        legal = b.possible_moves()
        if not legal:
            break
        b.move(*legal[rng.integers(len(legal))])
    players = {1: p1, 2: p2}
    for ply in range(400):
        if b.winning_player() or not b.possible_moves():
            break
        pl = players[b.current_player]
        counts, root = pl.mcts.search(b, add_noise=pl._ply < pl.explore_plies, rng=rng)
        if counts.sum() == 0:
            hits += 1
            print(f"\n=== HIT game {g} ply {ply} ===")
            print(f"  legal moves        {len(b.possible_moves())}")
            print(f"  terminal_value()   {terminal_value(b)}")
            print(f"  winning_player()   {b.winning_player()}")
            print(f"  root.is_terminal   {root.is_terminal}")
            print(f"  root.legal         {None if root.legal is None else len(root.legal)}")
            print(f"  root.child_N sum   {None if root.child_N is None else root.child_N.sum()}")
            print(f"  root.selected_act  {root.selected_action}")
            print(f"  policy sum/max     {counts.sum()} / {counts.max()}")
            print(f"  chips total        {sum(sum(r) for r in b.board)}")
            print("  board (chips:owner)")
            for y in range(b.size_y):
                print("   ", " ".join(f"{b.board[y][x]}:{b.player[y][x]}"
                                      for x in range(b.size_x)))
            break
        pl._ply += 1
        from alphazero.encoding import index_to_move
        b.move(*index_to_move(root.selected_action, b.size_x))
    print(f"game {g}: done (hits so far {hits})", flush=True)

print(f"\n{hits} hit(s) in {GAMES} games")
