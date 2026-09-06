"""Find the gumbel_c_scale where the TARGET stops being sharper than the SEARCH.

The completed policy is softmax(logits + sigma(q)) with
sigma(q) = (c_visit + max_visits) * c_scale * q. Too large and the target is
one-hot by construction, which teaches the net certainty the search never had.
We want the target's effective width to sit near the visit distribution's.
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
SCALES = (0.1, 0.05, 0.02, 0.01, 0.005)


def eff(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def main():
    net, _ = load_net(NET, 'cpu')
    ev = Evaluator(net, 'cpu')

    g = json.load(open(GAME))
    moves, n = g['moves'], g['size']
    positions = []
    for t in range(0, len(moves), 6):
        b = Board(n, n)
        for (x, y) in moves[:t]:
            b.move(int(x), int(y))
        if len(b.possible_moves()) >= 4:
            positions.append(b)

    print(f"{len(positions)} positions, 256 sims\n")
    print(f"{'c_scale':>8} | {'target width':>12} | {'visit width':>11} | {'target top':>10} | {'visit top':>9}")
    print("-" * 62)
    for cs in SCALES:
        cfg = Config(channels=64, res_blocks=4, num_simulations=256, gumbel_c_scale=cs)
        mcts = MCTS(ev, cfg)
        tw, vw, tt, vt = [], [], [], []
        for b in positions:
            pi, root = mcts.search(b, add_noise=False)
            v = np.asarray(root.child_N, dtype=np.float64)
            if v.sum() <= 0 or pi.sum() <= 0:
                continue
            v = v / v.sum()
            tw.append(eff(pi[pi > 0])); vw.append(eff(v))
            tt.append(pi.max()); vt.append(v.max())
        print(f"{cs:>8.3f} | {np.mean(tw):>12.2f} | {np.mean(vw):>11.2f} | "
              f"{np.mean(tt):>9.0%} | {np.mean(vt):>8.0%}", flush=True)


if __name__ == "__main__":
    main()
