"""Is the coverage-fix arm actually stronger than the old arm at the SAME round?

The AlphaBeta bar is 8 games per point (SE ~18%) and is played at 128 sims, where
even v174 scored only 8% — it cannot resolve anything this early. A direct match
between the two runs' versions can.
"""
import sys, time
sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')

from alphazero.config import Config
from alphazero.metrics import load_net
from alphazero.parallel import make_pool, match_parallel, net_arch_state
from alphazero.arena_eval import win_rate

A = sys.argv[1]
B = sys.argv[2]
GAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 40
SIMS = int(sys.argv[4]) if len(sys.argv) > 4 else 256
WORKERS = int(sys.argv[5]) if len(sys.argv) > 5 else 4


def main():
    na, _ = load_net(A, 'cpu')
    nb, _ = load_net(B, 'cpu')
    arch_a, state_a = net_arch_state(na)
    arch_b, state_b = net_arch_state(nb)
    cfg = Config(channels=64, res_blocks=4, num_simulations=SIMS,
                 eval_simulations=SIMS, match_random_plies=3)
    pool = make_pool(WORKERS)
    try:
        t = time.time()
        # ("az", arch, state) — a bare (arch, state) pair used to fall through
        # match_parallel's opponent dispatch into make_alphabeta(cfg, state), so
        # this script never played net B at all.
        wa, wb, dr = match_parallel(arch_a, state_a, ("az", arch_b, state_b), cfg,
                                    GAMES, 4242, pool, WORKERS)
        wr = win_rate(wa, wb, dr)
        se = (wr * (1 - wr) / GAMES) ** 0.5
        print(f"{A}\n  vs {B}\n  {wa}W-{wb}L-{dr}D = {wr:.0%} +/- {se:.0%} "
              f"for the first net  ({time.time()-t:.0f}s, {SIMS} sims)")
    finally:
        pool.close(); pool.join()


if __name__ == "__main__":
    main()
