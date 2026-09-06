"""Control: v39 vs the OLD C AlphaBeta through the repo's OWN match_parallel —
the exact code path the training gauntlet uses. If this also lands near 15%,
then vs_new_ab.py is fine and the gauntlet's 8-game 75% was noise.
"""
import sys, time
sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')
from dataclasses import replace
from alphazero.config import Config
from alphazero.metrics import load_net
from alphazero.parallel import make_pool, match_parallel, net_arch_state
from alphazero.arena_eval import win_rate


def main():
    games, workers = int(sys.argv[1]), int(sys.argv[2])
    net, meta = load_net("checkpoints_cover/best.pt", "cpu")
    arch, state = net_arch_state(net)
    print(f"v={meta.get('extra')} games={games}", flush=True)
    cfg = Config(match_random_plies=3, alphabeta_engine="c")
    pool = make_pool(workers)
    try:
        for sims in (128, 256):
            ecfg = replace(cfg, num_simulations=sims)
            t = time.time()
            wa, wb, dr = match_parallel(arch, state, ("alphabeta", 0.05), ecfg,
                                        games, 73000, pool, workers)
            print(f"  {sims:>4} sims vs old AB 0.05s: {wa}W-{wb}L-{dr}D = "
                  f"{win_rate(wa, wb, dr):.0%}  ({time.time()-t:.0f}s)", flush=True)
    finally:
        pool.close(); pool.join()


if __name__ == "__main__":
    main()
