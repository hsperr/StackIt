"""How good is the TEACHER? Does the self-play search agree with a strong AlphaBeta,
and does agreement improve with simulations?

The net can never be stronger than the search it is trained to imitate. If the
256-sim Gumbel search picks AlphaBeta's move only rarely, the training target is
the ceiling, not the net.

For ~60 positions drawn from net-vs-net games with random openings:
  * prior top-1 agreement with C-AlphaBeta @0.3s
  * Gumbel search SELECTED action agreement, at 128/256/512/1024 sims
  * Gumbel completed-policy ARGMAX agreement (that is what the net is trained on)
"""
import json
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')

from board import Board
from alphazero.config import Config
from alphazero.metrics import load_net
from alphazero.net import Evaluator
from alphazero.mcts_az import MCTS, terminal_value

BIN = '/Users/hsperr/code/youtube_coding/StackIt/c_engine/stackit'
NET = '/Users/hsperr/code/youtube_coding/StackIt/checkpoints_qblend/best.pt'
AB_SECS = 0.3
SIMS = (128, 256, 512, 1024)
N_POS = 60
N = 5


def positions(ev, rng):
    """Positions from net-guided games with random 3-ply openings."""
    cfg = Config(channels=64, res_blocks=4, num_simulations=64)
    mcts = MCTS(ev, cfg)
    out = []
    game = 0
    while len(out) < N_POS:
        game += 1
        b = Board(N, N)
        for _ in range(3):
            mv = b.possible_moves()
            if not mv:
                break
            x, y = mv[rng.integers(len(mv))]
            b.move(int(x), int(y))
        ply = 0
        while terminal_value(b) is None and ply < 60:
            if ply % 6 == 0 and len(b.possible_moves()) >= 4:
                out.append(b.copy())
                if len(out) >= N_POS:
                    break
            _, root = mcts.search(b, add_noise=True, rng=rng)
            b.move(*_idx(int(root.selected_action)))
            ply += 1
    return out


def _idx(a):
    from alphazero.encoding import index_to_move
    return index_to_move(a, N)


def ab_move(srv, b):
    cells = " ".join(f"{b.board[y][x]}:{b.player[y][x]}"
                     for y in range(N) for x in range(N))
    srv.stdin.write(f"{N} {N} {b.current_player} {AB_SECS} 40 {cells}\n")
    srv.stdin.flush()
    mv = json.loads(srv.stdout.readline()).get("move")
    return None if not mv else mv[0] * N + mv[1]


def main():
    rng = np.random.default_rng(7)
    net, _ = load_net(NET, 'cpu')
    ev = Evaluator(net, 'cpu')

    t0 = time.time()
    pos = positions(ev, rng)
    print(f"{len(pos)} positions in {time.time()-t0:.0f}s", flush=True)

    srv = subprocess.Popen([BIN, '--serve'], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, text=True, bufsize=1)
    refs, keep = [], []
    for b in pos:
        a = ab_move(srv, b)
        if a is not None:
            refs.append(a)
            keep.append(b)
    srv.stdin.close()
    print(f"{len(keep)} positions have an AlphaBeta reference move", flush=True)

    prior_hit = 0
    for b, a in zip(keep, refs):
        p, _ = ev.infer(b)
        prior_hit += int(np.argmax(p) == a)
    print(f"prior top-1 == AB@{AB_SECS}s : {prior_hit}/{len(keep)} = {prior_hit/len(keep):.0%}",
          flush=True)

    for s in SIMS:
        cfg = Config(channels=64, res_blocks=4, num_simulations=s)
        mcts = MCTS(ev, cfg)
        sel_hit = tgt_hit = 0
        t = time.time()
        for b, a in zip(keep, refs):
            pi, root = mcts.search(b, add_noise=False)
            sel_hit += int(int(root.selected_action) == a)
            tgt_hit += int(int(np.argmax(pi)) == a)
        n = len(keep)
        se = (0.25 / n) ** 0.5
        print(f"sims {s:5d}: played {sel_hit}/{n} = {sel_hit/n:.0%}   "
              f"target-argmax {tgt_hit}/{n} = {tgt_hit/n:.0%}   "
              f"(+/-{se:.0%}, {time.time()-t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
