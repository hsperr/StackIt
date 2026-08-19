"""Is the net over-confident, or is it TAUGHT to be over-confident?

Three distributions over the same positions:
  prior  - what the net says before searching
  visits - the raw MCTS visit counts, i.e. what the search actually believed
  target - the Gumbel completed policy, which is what gets trained on

If target is much sharper than visits, the net is being taught certainty the
search never had, and gumbel_c_scale / gumbel_c_visit is the knob. If target
tracks visits, the sharpness is coming from somewhere else.
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')

from board import Board
from alphazero.config import Config
from alphazero.metrics import load_net
from alphazero.net import Evaluator
from alphazero.mcts_az import MCTS

NET = '/Users/hsperr/code/youtube_coding/StackIt/checkpoints_qblend/best.pt'
GAME = '/Users/hsperr/code/youtube_coding/StackIt/checkpoints_qblend/last_game.json'


def ent(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


def eff(p):
    """Effective number of moves: exp(entropy). 1 = one-hot, N = uniform."""
    return float(np.exp(ent(p)))


def main():
    net, _ = load_net(NET, 'cpu')
    ev = Evaluator(net, 'cpu')
    cfg = Config(channels=64, res_blocks=4, num_simulations=256)
    mcts = MCTS(ev, cfg)

    g = json.load(open(GAME))
    moves, n = g['moves'], g['size']

    rows = []
    for t in range(0, len(moves), 6):
        b = Board(n, n)
        for (x, y) in moves[:t]:
            b.move(int(x), int(y))
        legal = b.possible_moves()
        if len(legal) < 4:
            continue

        prior, _ = ev.infer(b)
        pi, root = mcts.search(b, add_noise=False)
        visits = np.asarray(root.child_N, dtype=np.float64)
        if visits.sum() <= 0 or pi.sum() <= 0:
            continue
        rows.append((len(legal), eff(prior[prior > 0]), eff(visits / visits.sum()),
                     eff(pi[pi > 0]), float(pi.max()), float((visits / visits.sum()).max())))

    a = np.array(rows)
    print(f"positions: {len(a)}   mean legal moves: {a[:, 0].mean():.1f}")
    print()
    print("effective number of moves considered (exp of entropy; 1 = one-hot):")
    print(f"  net prior           : {a[:, 1].mean():.2f}")
    print(f"  MCTS visit counts   : {a[:, 2].mean():.2f}   <- what the search believed")
    print(f"  Gumbel TARGET (pi)  : {a[:, 3].mean():.2f}   <- what the net is trained on")
    print()
    print("mass on the single top move:")
    print(f"  MCTS visit counts   : {a[:, 5].mean():.0%}")
    print(f"  Gumbel TARGET (pi)  : {a[:, 4].mean():.0%}")


if __name__ == "__main__":
    main()
