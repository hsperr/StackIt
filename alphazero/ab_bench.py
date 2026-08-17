"""Background 'strong AlphaBeta' benchmark, decoupled from the training loop.

The trainer launches this as a detached subprocess for the occasional champion
(every cfg.ab_bench_every accepted models). It plays the champion net against
AlphaBeta with a big think budget — cfg.ab_bench_think seconds/move for *both*
sides (the net searches by wall-clock, AlphaBeta by the same budget) — across a
few CPU cores, then appends one summary row to cfg.ab_bench_file. It never
blocks training; a lockfile stops two benchmarks from overlapping.

Run indirectly (the trainer does this):
    python3 -m alphazero.ab_bench --weights versions/v42.pt \
        --champion-iter 42 --train-iter 57 --cfg '<json>'
"""
import os
import sys
import json
import time
import argparse
from dataclasses import replace

from .config import Config
from .parallel import make_pool, match_parallel
from .arena_eval import win_rate


def benchmark_running(cfg):
    """True iff a LIVE ab_bench process holds the lock. A lock left by a crashed
    process (pid gone) or a garbage lock is cleared and treated as not-running, so
    one dead benchmark can't suppress every future benchmark indefinitely."""
    lock = cfg.ab_bench_file + ".lock"
    try:
        with open(lock) as f:
            pid = int(f.read().strip())
    except FileNotFoundError:
        return False
    except ValueError:
        _rm(lock)                    # unreadable lock — treat as stale
        return False
    try:
        os.kill(pid, 0)              # probe: does the process still exist?
    except ProcessLookupError:
        _rm(lock)                    # stale: owner is gone
        return False
    except PermissionError:
        return True                  # exists, owned by someone else
    return True


def _rm(path):
    try:
        os.remove(path)
    except OSError:
        pass


def run_bench(cfg, weights_path, champion_iter, train_iter):
    import torch
    d = torch.load(weights_path, map_location="cpu", weights_only=False)
    arch, state = d["arch"], d["state"]
    # net side searches by wall-clock; AlphaBeta gets the same seconds/move
    bcfg = replace(cfg, az_time_budget=cfg.ab_bench_think)
    pool = make_pool(cfg.ab_bench_workers)
    try:
        t0 = time.time()
        wa, wb, dr = match_parallel(arch, state, ("alphabeta", cfg.ab_bench_think),
                                    bcfg, cfg.ab_bench_games, 4242 + champion_iter,
                                    pool, cfg.ab_bench_workers)
        row = {"champion_iter": champion_iter, "train_iter": train_iter,
               "think": cfg.ab_bench_think, "games": cfg.ab_bench_games,
               "wins": wa, "losses": wb, "draws": dr,
               "winrate": round(win_rate(wa, wb, dr), 3),
               "sec": round(time.time() - t0, 1)}
        with open(cfg.ab_bench_file, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[ab-bench] v{champion_iter}: {row['winrate']:.2f} "
              f"({wa}-{wb}-{dr}) in {row['sec']}s")
    finally:
        pool.close()
        pool.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--champion-iter", type=int, required=True)
    ap.add_argument("--train-iter", type=int, required=True)
    ap.add_argument("--cfg", required=True, help="JSON dump of the Config")
    a = ap.parse_args()
    cfg = Config(**json.loads(a.cfg))

    benchmark_running(cfg)           # clear a stale lock left by a crashed process
    lock = cfg.ab_bench_file + ".lock"
    try:
        # atomic create-or-fail: if two launches race past the check above, only
        # one wins the O_EXCL create; the loser backs off instead of double-running.
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return                       # another LIVE benchmark holds the lock — skip
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        run_bench(cfg, a.weights, a.champion_iter, a.train_iter)
    finally:
        _rm(lock)


if __name__ == "__main__":
    main()
