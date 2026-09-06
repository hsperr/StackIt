"""Is a FIXED-DEPTH C AlphaBeta load-invariant and deterministic?

The serve protocol is "<sx> <sy> <side> <secs> <max_depth> <cells...>". Give it
a huge <secs> and a small <max_depth> and the wall clock never fires:
time_over_exact() stays false, iterative deepening runs depth 0..max_depth, and
result.best_move is always committed. Strength then depends on nothing but the
depth -- not on how busy the machine is, which is the whole defect in the
0.05 s/move bar.

Checks: same move every time, at every concurrency, and what each depth costs.
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, "/Users/hsperr/code/youtube_coding/StackIt")
from board import Board

ROOT = "/Users/hsperr/code/youtube_coding/StackIt"
BIN = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/c_engine/stackit"
HUGE = 1e9          # secs: never fires

# 12 positions from real random openings of varying length
POS = []
for s in range(12):
    rng = np.random.default_rng(1000 + s)
    b = Board(5, 5)
    for _ in range(3 + 2 * s):
        mv = b.possible_moves()
        if not mv:
            break
        b.move(*mv[rng.integers(len(mv))])
    POS.append(" ".join(f"{b.board[y][x]}:{b.player[y][x]}" for y in range(5)
                        for x in range(5)) if True else None)
    POS[-1] = (b.current_player, POS[-1])


def ask(depth, k):
    def one(_):
        pr = subprocess.Popen([BIN, "--serve"], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True, bufsize=1)
        try:
            out = []
            for side, cells in POS:
                pr.stdin.write(f"5 5 {side} {HUGE} {depth} {cells}\n"); pr.stdin.flush()
                out.append(json.loads(pr.stdout.readline()))
            return out
        finally:
            pr.kill()
    with ThreadPoolExecutor(k) as ex:
        return list(ex.map(one, range(k)))


print(f"binary={os.path.basename(BIN)}  load={os.getloadavg()[0]:.1f}  {len(POS)} positions")
print("depth  idle move set        K=10 agrees   nulls   secs/pos (median, K=10)")
base = None
for depth in (4, 5, 6, 7, 8):
    t = time.time()
    r1 = ask(depth, 1)[0]
    solo = time.time() - t
    rk = ask(depth, 10)
    moves1 = [tuple(r["move"]) if r["move"] else None for r in r1]
    agree = all([tuple(r["move"]) if r["move"] else None for r in run] == moves1 for run in rk)
    nulls = sum(r["move"] is None for run in rk for r in run)
    secs = sorted(r["secs"] for run in rk for r in run)
    print(f"{depth:>5}  {str(moves1[:3]):<20} {str(agree):>11}   {nulls:>5}   "
          f"{secs[len(secs)//2]:.4f}  (solo total {solo:.2f}s)", flush=True)
