"""Is v174's policy too narrow, and does it discard the moves AlphaBeta likes?

For a set of real positions:
  - how many moves carry 90% of the policy mass (the net's effective width)
  - what RANK the C-AlphaBeta best move has in the net's policy ordering
  - whether the net's own Gumbel MCTS ends up playing AlphaBeta's move
Single process, no pool — deliberately light, a match is running alongside.
"""
import json
import subprocess
import sys

import numpy as np

sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')

from board import Board
from alphazero.config import Config
from alphazero.metrics import load_net
from alphazero.net import Evaluator
from alphazero.mcts_az import MCTS

BIN = '/Users/hsperr/code/youtube_coding/StackIt/c_engine/stackit'
GAME = '/Users/hsperr/code/youtube_coding/StackIt/checkpoints_qblend/last_game.json'
AB_SECS = 0.05


def main():
    net, _ = load_net('/Users/hsperr/code/youtube_coding/StackIt/checkpoints_qblend/best.pt', 'cpu')
    ev = Evaluator(net, 'cpu')
    cfg = Config(channels=64, res_blocks=4, num_simulations=256)
    mcts = MCTS(ev, cfg)

    g = json.load(open(GAME))
    moves, n = g['moves'], g['size']
    srv = subprocess.Popen([BIN, '--serve'], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, text=True, bufsize=1)

    widths, ranks, mcts_hits, legal_counts = [], [], 0, []
    for t in range(0, len(moves), 8):
        b = Board(n, n)
        for (x, y) in moves[:t]:
            b.move(int(x), int(y))
        legal = b.possible_moves()
        if len(legal) < 4:
            continue

        priors, _ = ev.infer(b)
        p = priors[priors > 0]
        order = np.sort(p)[::-1]
        widths.append(int(np.searchsorted(np.cumsum(order), 0.90) + 1))
        legal_counts.append(len(legal))

        cells = " ".join(f"{b.board[y][x]}:{b.player[y][x]}"
                         for y in range(n) for x in range(n))
        srv.stdin.write(f"{n} {n} {b.current_player} {AB_SECS} 40 {cells}\n")
        srv.stdin.flush()
        ab = json.loads(srv.stdout.readline()).get("move")
        if not ab:
            continue
        ab_idx = ab[0] * n + ab[1]

        # rank of AlphaBeta's move in the net's policy (1 = the net's top choice)
        ranks.append(int((priors > priors[ab_idx]).sum()) + 1)

        pi, root = mcts.search(b, add_noise=False)
        if int(root.selected_action) == ab_idx:
            mcts_hits += 1

    srv.stdin.close()
    w, r = np.array(widths), np.array(ranks)
    print(f"positions: {len(w)}   median legal moves: {int(np.median(legal_counts))}")
    print(f"policy width (moves holding 90% of mass): median {int(np.median(w))}, "
          f"mean {w.mean():.1f}, max {w.max()}")
    print(f"AlphaBeta's move ranked in the net's policy: median {int(np.median(r))}, "
          f"mean {r.mean():.1f}")
    print(f"  AB move was the net's #1 choice : {(r == 1).sum()}/{len(r)}")
    print(f"  AB move inside the net's top 3  : {(r <= 3).sum()}/{len(r)}")
    print(f"  AB move outside the net's top 8 : {(r > 8).sum()}/{len(r)}")
    print(f"net's 256-sim MCTS played AB's move: {mcts_hits}/{len(r)}")


if __name__ == "__main__":
    main()
