"""How strong are our nets, measured against the C AlphaBeta?

The __main__ guard is load-bearing: alphazero.parallel uses a SPAWN context, so
every worker re-imports this file. Without the guard each worker builds its own
pool and the thing fork bombs.
"""
import sys
import time

sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')

from alphazero.config import Config
from alphazero.metrics import load_net
from alphazero.parallel import make_pool, match_parallel, net_arch_state
from alphazero.arena_eval import win_rate

NETS = (("v174 old arm 64x4", "checkpoints_qblend/best.pt"),
        ("v30  new arm 96x5", "checkpoints_big/best.pt"))
BUDGETS = (0.05, 0.3)
GAMES = 20


def main():
    cfg = Config(num_simulations=128, eval_simulations=128,
                 match_random_plies=3, alphabeta_engine="c")
    pool = make_pool(8)
    try:
        for label, path in NETS:
            net, _ = load_net(path, "cpu")
            arch, state = net_arch_state(net)
            for budget in BUDGETS:
                t = time.time()
                wa, wb, dr = match_parallel(arch, state, ("alphabeta", budget),
                                            cfg, GAMES, 4242, pool, 8)
                print(f"{label} vs C-AlphaBeta {budget}s/move: "
                      f"{wa}W-{wb}L-{dr}D = {win_rate(wa, wb, dr):.0%}  "
                      f"({time.time() - t:.0f}s)", flush=True)
    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
