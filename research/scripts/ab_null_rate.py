"""Does the C AlphaBeta return NO MOVE under load?

ab_load.py measured depth and nodes vs concurrency and concluded "load is not
the explanation". It never looked at the one field that matters: whether the
engine answers {"move":null}. search_best_move() only commits result.best_move
*after* checking time_over_exact(), and it takes t0 BEFORE the 100 MB
`memset(g_tt, ...)` that clears the transposition table on every call
(TT_BITS=22, 24-byte entries). If that memset plus one depth-0 pass overruns the
budget, best_move stays -1 and the engine answers null.

arena_eval/_match_chunk treats a null move as `break`, and game_winner() then
scores the half-finished position by box count. So a null-ing engine does not
lose games -- it hands them to the net.

Prints, per concurrency level: null rate, median depth, median nodes, and the
system load it ran under.
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = "/Users/hsperr/code/youtube_coding/StackIt"
BIN = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/c_engine/stackit"
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 30

# a mid-game 5x5 position, side 1 to move
CELLS = " ".join("1:1" if i in (0, 6, 12, 18, 24) else
                 ("1:2" if i in (4, 8, 16, 20) else "0:0") for i in range(25))
REQ = f"5 5 1 {BUDGET} 40 {CELLS}\n"


def one(_):
    pr = subprocess.Popen([BIN, "--serve"], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True, bufsize=1)
    try:
        out = []
        for _ in range(REPS):
            pr.stdin.write(REQ); pr.stdin.flush()
            out.append(json.loads(pr.stdout.readline()))
        return out
    finally:
        pr.kill()


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


print(f"binary={os.path.basename(BIN)} budget={BUDGET}s reps={REPS}/engine")
print(f"load at start: {os.getloadavg()}")
for k in (1, 2, 4, 8, 12):
    t = time.time()
    with ThreadPoolExecutor(k) as ex:
        rs = [r for chunk in ex.map(one, range(k)) for r in chunk]
    nulls = sum(1 for r in rs if r.get("move") is None)
    print(f"{k:>3} concurrent engines: NULL {nulls:>4}/{len(rs):<4} = {nulls/len(rs):>5.0%}   "
          f"median depth {med([r['depth'] for r in rs]):>3}   "
          f"median nodes {med([r['nodes'] for r in rs]):>9,}   "
          f"median secs {med([r['secs'] for r in rs]):.4f}   "
          f"load {os.getloadavg()[0]:.1f}  ({time.time()-t:.0f}s)", flush=True)
