"""Replay the trainer's AlphaBeta gauntlet with per-game instrumentation.

The trainer's book shows, for every iteration whose eval block took 0.3 s
(12/24/33/36/39/42), EXACTLY 4 net wins out of 8 -- the 4 even-numbered games,
which are precisely the games where AlphaBeta is the first to move after the 3
random opening plies. That is an abort signature, not a strength signature, so
this records what actually happens: plies played, who failed, and how.

Same code as alphazero/parallel._match_chunk, plus counters. Reports the load it
ran under, because the C engine plays to a wall clock.
"""
import json, os, sys, time
from multiprocessing import get_context

import numpy as np
import torch

ROOT = "/Users/hsperr/code/youtube_coding/StackIt"
sys.path.insert(0, ROOT)
_CTX = get_context("spawn")


def _init():
    torch.set_num_threads(1)


def _chunk(payload):
    torch.set_num_threads(1)
    from board import Board
    from alphazero.config import Config
    from alphazero.net import StackNet
    from alphazero.net import Evaluator
    from alphazero.selfplay import game_winner
    from alphazero.arena_eval import AZPlayer, make_alphabeta

    arch, state, budget, specs, cfg_dict = payload
    cfg = Config(**cfg_dict)
    net = StackNet(arch["board_size"], arch["channels"], arch["res_blocks"])
    net.load_state_dict(state)
    net.eval()
    ev = Evaluator(net, torch.device("cpu"))

    out = []
    for first_is_a, seed in specs:
        rng = np.random.default_rng(int(seed))
        pa, pb = AZPlayer(ev, cfg), make_alphabeta(cfg, budget)
        pa.reset(); pb.reset()
        p1, p2 = (pa, pb) if first_is_a else (pb, pa)
        players = {1: p1, 2: p2}
        b = Board(cfg.board_size, cfg.board_size)
        for _ in range(getattr(cfg, "match_random_plies", 0)):
            legal = b.possible_moves()
            if not legal:
                break
            b.move(*legal[rng.integers(len(legal))])
        plies, fail = 0, None
        t0 = time.time()
        for _ in range(cfg.max_game_plies):
            if b.winning_player() or not b.possible_moves():
                break
            who = players[b.current_player]
            mv = who.move(b, rng)
            if mv is None or tuple(mv) not in b.possible_moves():
                fail = ("net" if who is pa else "ab") + ("/none" if mv is None else "/illegal")
                break
            b.move(*mv)
            plies += 1
        w = game_winner(b)
        res = "draw" if w == 0 else ("a" if (p1 if w == 1 else p2) is pa else "b")
        out.append(dict(first_is_a=first_is_a, seed=int(seed), plies=plies, fail=fail,
                        res=res, secs=round(time.time() - t0, 2),
                        boxes=(b.boxes_for(1), b.boxes_for(2))))
    return out


def _split(items, n):
    return [c for c in (items[i::n] for i in range(n)) if c]


def main():
    from dataclasses import replace
    from alphazero.config import Config
    from alphazero.metrics import load_net
    from alphazero.parallel import net_arch_state
    from alphazero.arena_eval import win_rate

    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_cover/versions/v39.pt"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 3973000     # v39: 0 + 39*100000 + 73000
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    reps = int(sys.argv[5]) if len(sys.argv) > 5 else 3

    net, _ = load_net(os.path.join(ROOT, ckpt), "cpu")
    arch, state = net_arch_state(net)
    cfg = replace(Config(match_random_plies=3, alphabeta_engine="c"),
                  num_simulations=128)
    print(f"ckpt={ckpt} seed={seed} games={games} workers={workers} "
          f"load={os.getloadavg()[0]:.1f}", flush=True)
    pool = _CTX.Pool(workers, initializer=_init)
    try:
        for rep in range(reps):
            specs = [(g % 2 == 0, seed + g) for g in range(games)]
            payloads = [(arch, state, 0.05, c, cfg.to_dict()) for c in _split(specs, workers)]
            t = time.time()
            rows = [r for res in pool.map(_chunk, payloads) for r in res]
            wa = sum(r["res"] == "a" for r in rows)
            wb = sum(r["res"] == "b" for r in rows)
            dr = sum(r["res"] == "draw" for r in rows)
            fails = [r["fail"] for r in rows if r["fail"]]
            print(f"rep{rep}: {wa}W-{wb}L-{dr}D = {win_rate(wa, wb, dr):.3f}  "
                  f"{time.time()-t:.1f}s  load={os.getloadavg()[0]:.1f}  "
                  f"median_plies={sorted(r['plies'] for r in rows)[len(rows)//2]}  "
                  f"aborts={len(fails)} {sorted(set(fails))}", flush=True)
            for r in sorted(rows, key=lambda r: r["seed"]):
                print("   ", json.dumps(r), flush=True)
    finally:
        pool.close(); pool.join()


if __name__ == "__main__":
    main()
