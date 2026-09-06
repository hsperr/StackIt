"""How steep is the net's score against a wall-clock AlphaBeta as its budget moves?
If a 2-4x throttle turns 10% into 75%, then every winrate_vs_ab_light number is
really a measure of how busy the machine was."""
import sys, time
sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')
from dataclasses import replace
from alphazero.config import Config
from alphazero.metrics import load_net
from alphazero.parallel import make_pool, match_parallel, net_arch_state
from alphazero.arena_eval import win_rate

games, workers = int(sys.argv[1]), int(sys.argv[2])
net, _ = load_net("checkpoints_cover/versions/v39.pt", "cpu")
arch, state = net_arch_state(net)
cfg = replace(Config(match_random_plies=3, alphabeta_engine="c"), num_simulations=128)
if __name__ == "__main__":
    pool = make_pool(workers)
    try:
        for budget in (0.005, 0.0125, 0.025, 0.05):
            t = time.time()
            wa, wb, dr = match_parallel(arch, state, ("alphabeta", budget), cfg,
                                        games, 3973000, pool, workers)
            print(f"  AB budget {budget:>6}s: net {wa}W-{wb}L-{dr}D = "
                  f"{win_rate(wa, wb, dr):.0%}  ({time.time()-t:.0f}s)", flush=True)
    finally:
        pool.close(); pool.join()
