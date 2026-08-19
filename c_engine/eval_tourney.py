"""Round-robin between AlphaBeta engines that differ ONLY in board_eval.

board.c carries every candidate behind `-DEVAL_VARIANT=n`, so each engine is the
same search, the same move ordering and the same quiescence with one thing
changed. This script builds them into a scratch directory and plays them off.

    python c_engine/eval_tourney.py depth 40        # fixed depth 5
    python c_engine/eval_tourney.py time  60        # fixed 0.05s/move
    python c_engine/eval_tourney.py slow  30        # fixed 0.30s/move
    python c_engine/eval_tourney.py depth 40 3 0,4  # only these variants

Run fixed DEPTH first: it asks "is the eval smarter?" without letting a slower
eval lose on the clock. Then run fixed TIME, which asks the question that
actually matters — does it pay for itself? A smarter eval that costs a ply can
still come out behind.

It never touches `c_engine/stackit`. Rebuilding that binary mid-training-run
changes the benchmark opponent under a live run, which makes the AlphaBeta line
on the dashboard meaningless.
"""
import itertools
import json
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from board import Board  # noqa: E402

N = 5
OPEN_PLIES = 3           # random opening plies, so a "40-game match" is 40 games
MAX_PLIES = 300
VARIANTS = {0: "chips (pre-2026-08-19)", 1: "territory", 2: "terr+threat (2 pass)",
            3: "chips+threat", 4: "terr+threat (default)"}
BUILD_DIR = os.path.join(tempfile.gettempdir(), "stackit_eval_variants")


def build(which):
    os.makedirs(BUILD_DIR, exist_ok=True)
    out = {}
    for v in which:
        path = os.path.join(BUILD_DIR, f"stackit_e{v}")
        cmd = ["clang", "-std=c99", "-O3", "-Wall", "-Wextra", f"-DEVAL_VARIANT={v}",
               "-o", path, "board.c", "search.c", "main.c"]
        subprocess.run(cmd, cwd=HERE, check=True)
        out[f"e{v} {VARIANTS[v]}"] = path
    return out


def ask(srv, b, secs, depth):
    cells = " ".join(f"{b.board[y][x]}:{b.player[y][x]}"
                     for y in range(N) for x in range(N))
    srv.stdin.write(f"{N} {N} {b.current_player} {secs} {depth} {cells}\n")
    srv.stdin.flush()
    return json.loads(srv.stdout.readline()).get("move")


def winner(b):
    """Arena rules: domination first, else box count."""
    w = b.winning_player()
    if w:
        return w
    b1, b2 = b.boxes_for(1), b.boxes_for(2)
    return 1 if b1 > b2 else 2 if b2 > b1 else 0


def one_game(job):
    bin_a, bin_b, seed, a_is_p1, secs, depth = job
    rng = np.random.default_rng(seed)
    procs = [subprocess.Popen([p, "--serve"], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True, bufsize=1)
             for p in (bin_a, bin_b)]
    try:
        b = Board(N, N)
        for _ in range(OPEN_PLIES):
            mv = b.possible_moves()
            if not mv:
                break
            x, y = mv[rng.integers(len(mv))]
            b.move(int(x), int(y))
        for _ in range(MAX_PLIES):
            if b.winning_player() or not b.possible_moves():
                break
            a_moves = (b.current_player == 1) == a_is_p1
            mv = ask(procs[0 if a_moves else 1], b, secs, depth)
            if not mv:
                break
            b.move(int(mv[0]), int(mv[1]))
        w = winner(b)
        if w == 0:
            return 0
        return 1 if ((w == 1) == a_is_p1) else -1
    finally:
        for p in procs:
            try:
                p.stdin.close()
                p.wait(timeout=2)
            except Exception:
                p.kill()


def run(engines, games, secs, depth, workers, label):
    names = list(engines)
    jobs, pairs = [], []
    for a, c in itertools.combinations(names, 2):
        pairs.append((a, c))
        # colours alternate across games so neither engine gets the first move twice
        jobs += [(engines[a], engines[c], 1000 + g, g % 2 == 0, secs, depth)
                 for g in range(games)]
    with Pool(workers) as pool:
        res = pool.map(one_game, jobs)

    print(f"\n=== {label} ===")
    score = {n: 0.0 for n in names}
    played = {n: 0 for n in names}
    for i, (a, c) in enumerate(pairs):
        r = res[i * games:(i + 1) * games]
        w = sum(1 for x in r if x == 1)
        l = sum(1 for x in r if x == -1)
        d = sum(1 for x in r if x == 0)
        wr = (w + 0.5 * d) / games
        se = (wr * (1 - wr) / games) ** 0.5
        print(f"  {a:>26} vs {c:<26} {w:2d}W-{l:2d}L-{d:2d}D = {wr:.0%} +/- {se:.0%}")
        score[a] += w + 0.5 * d
        score[c] += l + 0.5 * d
        played[a] += games
        played[c] += games
    if len(names) > 2:
        print("  --- overall ---")
        for n in sorted(names, key=lambda k: -score[k] / played[k]):
            print(f"  {n:>26}: {score[n]/played[n]:.0%}  ({score[n]:.1f}/{played[n]})")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "depth"
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    which = ([int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4
             else sorted(VARIANTS))
    engines = build(which)
    if mode == "depth":
        run(engines, games, 100.0, 5, workers, f"fixed depth 5, {games} games/pair")
    elif mode == "time":
        run(engines, games, 0.05, 40, workers, f"fixed 0.05s/move, {games} games/pair")
    else:
        run(engines, games, 0.3, 40, workers, f"fixed 0.30s/move, {games} games/pair")


if __name__ == "__main__":
    main()
