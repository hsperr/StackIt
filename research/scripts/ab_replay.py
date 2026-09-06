"""Replay a training gauntlet's AlphaBeta match exactly: same checkpoint, same
seed (cfg.seed + it*100000 + 73000), same sims, same code path."""
import sys, time
sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')
from dataclasses import replace
from alphazero.config import Config
from alphazero.metrics import load_net
from alphazero.parallel import make_pool, match_parallel, net_arch_state
from alphazero.arena_eval import win_rate


def main():
    it, games, workers = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    seed = 0 + it * 100000 + 73000
    net, _ = load_net(f"checkpoints_cover/versions/v{it}.pt", "cpu")
    arch, state = net_arch_state(net)
    cfg = replace(Config(match_random_plies=3, alphabeta_engine="c"),
                  num_simulations=128)
    pool = make_pool(workers)
    try:
        for rep in range(3):                       # same seed 3x -> is it deterministic?
            t = time.time()
            wa, wb, dr = match_parallel(arch, state, ("alphabeta", 0.05), cfg,
                                        games, seed, pool, workers)
            print(f"  v{it} seed={seed} rep{rep}: {wa}W-{wb}L-{dr}D = "
                  f"{win_rate(wa, wb, dr):.0%}  ({time.time()-t:.0f}s)", flush=True)
    finally:
        pool.close(); pool.join()


if __name__ == "__main__":
    main()
