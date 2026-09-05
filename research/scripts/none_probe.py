"""Replay the exact position the abort guard captured and find where the zero
policy comes from.

    player 2 to move, 23 legal moves, winning_player=0, terminal_value=None
    board  [[3,0,0,0,0],[0,0,0,0,3],[0,0,0,0,0],[0,0,0,3,0],[0,0,0,0,0]]
    owner  [[1,0,0,0,0],[0,0,0,0,2],[0,0,0,0,0],[0,0,0,1,0],[0,0,0,0,0]]
"""
import sys
import numpy as np

sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')

from board import Board
from alphazero.config import Config
from alphazero.metrics import load_net
from alphazero.net import Evaluator
from alphazero.mcts_az import MCTS, Node, terminal_value
from alphazero.encoding import legal_mask, encode

CH = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints_cover/versions/v3.pt'
SIMS = int(sys.argv[2]) if len(sys.argv) > 2 else 128

CHIPS = [[3, 0, 0, 0, 0], [0, 0, 0, 0, 3], [0, 0, 0, 0, 0], [0, 0, 0, 3, 0], [0, 0, 0, 0, 0]]
OWNER = [[1, 0, 0, 0, 0], [0, 0, 0, 0, 2], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]

b = Board(5, 5)
for y in range(5):
    for x in range(5):
        b.board[y][x] = CHIPS[y][x]
        b.player[y][x] = OWNER[y][x]
b.current_player = 2

print(f"legal moves      {len(b.possible_moves())}")
print(f"terminal_value   {terminal_value(b)}")
print(f"legal_mask sum   {legal_mask(b).sum()}")

net, _ = load_net(CH, 'cpu')
ev = Evaluator(net, 'cpu')
priors, v = ev.infer(b)
print(f"\nevaluator: value {v:.4f}  priors sum {priors.sum():.6f}  "
      f"max {priors.max():.6f}  nan {np.isnan(priors).any()}")

node = Node(b.copy())
tv = node.expand(ev)
print(f"\nnode.expand -> {tv}")
print(f"  is_terminal {node.is_terminal}")
print(f"  legal       {None if node.legal is None else len(node.legal)}")
print(f"  priors      {None if node.priors is None else (node.priors.sum(), node.priors.min())}")

cfg = Config(channels=64, res_blocks=4, num_simulations=SIMS, eval_simulations=SIMS)
mcts = MCTS(ev, cfg)
for trial in range(5):
    pi, root = mcts.search(b.copy(), add_noise=True, rng=np.random.default_rng(trial))
    print(f"\ntrial {trial}: pi.sum {pi.sum()}  pi.max {pi.max()}  "
          f"nan {np.isnan(pi).any()}  selected {root.selected_action}  "
          f"is_terminal {root.is_terminal}  legal {None if root.legal is None else len(root.legal)}")
    if pi.sum() == 0:
        print("  ^^ ZERO POLICY REPRODUCED")
