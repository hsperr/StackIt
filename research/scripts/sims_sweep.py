"""Does raising SIMS actually help v174, or was that 15%->30% just noise?

The c_puct sweep accidentally ran at 256 sims while the earlier baseline ran at
128, and the score doubled. 20 games has a standard error of ~11%, so that is
suggestive, not proof. This runs 40 games per point at a fixed c_puct so sims is
the only thing changing.
"""
import sys
import time

sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')

from alphazero.config import Config
from alphazero.metrics import load_net
from alphazero.parallel import make_pool, match_parallel, net_arch_state
from alphazero.arena_eval import win_rate

NET = '/Users/hsperr/code/youtube_coding/StackIt/checkpoints_qblend/best.pt'
SIMS = (128, 256, 512, 1024)
GAMES = 40
AB_SECS = 0.05


def main():
    net, _ = load_net(NET, 'cpu')
    arch, state = net_arch_state(net)
    pool = make_pool(8)
    try:
        for s in SIMS:
            cfg = Config(channels=64, res_blocks=4, c_puct=1.5,
                         num_simulations=s, eval_simulations=s,
                         match_random_plies=3, alphabeta_engine="c")
            t = time.time()
            wa, wb, dr = match_parallel(arch, state, ("alphabeta", AB_SECS), cfg,
                                        GAMES, 909, pool, 8)
            wr = win_rate(wa, wb, dr)
            se = (wr * (1 - wr) / GAMES) ** 0.5
            print(f"sims {s:5d}: {wa}W-{wb}L-{dr}D = {wr:.0%} +/- {se:.0%}  "
                  f"({time.time() - t:.0f}s)", flush=True)
    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
