"""Does tree reuse inflate sigma's max-visit term?

mctx builds a fresh tree every move, so max_N is bounded by the simulation
budget and sigma = (c_visit + max_N) * c_scale is bounded too. With tree reuse a
root arrives carrying visits from earlier plies, so max_N — and therefore the
sharpness of the training target — runs past what the algorithm intends.

Plays one self-play game and prints, per recorded move, the max root visit count
INCLUDING inherited visits vs the visits this search actually spent.
"""
import sys
import numpy as np

sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')

from board import Board
from alphazero.config import Config
from alphazero.metrics import load_net
from alphazero.net import Evaluator
from alphazero.mcts_az import MCTS, terminal_value
from alphazero.encoding import index_to_move

NET = '/Users/hsperr/code/youtube_coding/StackIt/checkpoints_qblend/best.pt'


def main():
    net, _ = load_net(NET, 'cpu')
    ev = Evaluator(net, 'cpu')
    cfg = Config(channels=64, res_blocks=4, num_simulations=256)
    mcts = MCTS(ev, cfg)
    rng = np.random.default_rng(3)

    b = Board(5, 5)
    for _ in range(3):
        mv = b.possible_moves()
        x, y = mv[rng.integers(len(mv))]
        b.move(x, y)

    reuse = None
    rows = []
    for ply in range(3, 60):
        if terminal_value(b) is not None:
            break
        record = rng.random() < cfg.pcr_prob
        budget = cfg.num_simulations if record else cfg.pcr_fast_sims
        before = None if reuse is None else reuse.child_N.copy()
        pi, root = mcts.search(b, add_noise=True, max_sims=budget,
                               reuse_root=reuse, rng=rng)
        if pi.sum() == 0:
            break
        base = before if (before is not None and root is reuse) else np.zeros_like(root.child_N)
        if record:
            rows.append((budget, float((root.child_N - base).max()), float(root.child_N.max())))
        a = root.selected_action
        li = np.where(root.legal == a)[0]
        reuse = root.children.get(int(li[0])) if len(li) else None
        b.move(*index_to_move(a, 5))

    print(f"{'budget':>7} {'maxN(this search)':>18} {'maxN(with reuse)':>17} "
          f"{'sigma span own':>15} {'sigma span reuse':>17}")
    for bud, own, tot in rows:
        print(f"{bud:>7} {own:>18.0f} {tot:>17.0f} "
              f"{(50+own)*0.1:>15.2f} {(50+tot)*0.1:>17.2f}")
    if rows:
        o = np.array([r[1] for r in rows]); t = np.array([r[2] for r in rows])
        print(f"\nmedian inflation of the sigma multiplier: "
              f"{np.median((50+t)/(50+o)):.2f}x")


if __name__ == "__main__":
    main()
