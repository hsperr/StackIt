"""Test fix B on the EXISTING v174 net — no retraining.

c_puct controls how far interior-node PUCT strays from the prior. If the plateau
really is a narrow-prior trap, raising it should let the search reach the moves
the policy nearly rules out, and the score against the C AlphaBeta should move.
If nothing moves, the diagnosis is wrong and we saved ourselves a training run.
"""
import sys
import time

sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')

from alphazero.config import Config
from alphazero.metrics import load_net
from alphazero.parallel import make_pool, match_parallel, net_arch_state
from alphazero.arena_eval import win_rate

NET = '/Users/hsperr/code/youtube_coding/StackIt/checkpoints_qblend/best.pt'
C_PUCTS = (1.5, 3.0, 5.0, 8.0)
GAMES = 20
AB_SECS = 0.05


def main():
    net, _ = load_net(NET, 'cpu')
    arch, state = net_arch_state(net)
    pool = make_pool(8)
    try:
        for cp in C_PUCTS:
            cfg = Config(channels=64, res_blocks=4, c_puct=cp,
                         num_simulations=256, eval_simulations=256,
                         match_random_plies=3, alphabeta_engine="c")
            t = time.time()
            wa, wb, dr = match_parallel(arch, state, ("alphabeta", AB_SECS), cfg,
                                        GAMES, 4242, pool, 8)
            print(f"c_puct {cp:4.1f}: v174 vs C-AB {AB_SECS}s = "
                  f"{wa}W-{wb}L-{dr}D = {win_rate(wa, wb, dr):.0%}  "
                  f"({time.time() - t:.0f}s)", flush=True)
    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
