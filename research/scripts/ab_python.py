"""Is the training gauntlet's 'alphabeta' actually the Python engine?
make_alphabeta() falls back to it (with a print that a buffered log would swallow).
The Python AlphaBeta gets ~1/50th the nodes at the same 0.05s, so it is a far
weaker opponent -- which would explain a 75% gauntlet score next to 0% vs C."""
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
if __name__ == "__main__":
    pool = make_pool(workers)
    try:
        for engine in ("python", "c"):
            cfg = replace(Config(match_random_plies=3, alphabeta_engine=engine),
                          num_simulations=128)
            t = time.time()
            wa, wb, dr = match_parallel(arch, state, ("alphabeta", 0.05), cfg,
                                        games, 3973000, pool, workers)
            print(f"  v39 vs {engine:>6} AlphaBeta 0.05s: {wa}W-{wb}L-{dr}D = "
                  f"{win_rate(wa, wb, dr):.0%}  ({time.time()-t:.0f}s)", flush=True)
    finally:
        pool.close(); pool.join()
