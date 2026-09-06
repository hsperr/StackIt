"""Is it the FIRST search of a fresh engine process that returns null?

search_best_move() lazily malloc()s a 100 MB transposition table
(TT_BITS=22 * 24 B) and memset()s it -- both INSIDE the timed budget, because
t0 is taken before them. On the first call the pages are untouched, so the
memset is a 100 MB first-touch fault storm. Every later call only re-memsets
already-resident pages.

Spawns K fresh engines at once (what a training match does: 8 games spread over
10 spawn-pool workers, each building its own CAlphaBetaPlayer._shared Popen) and
reports the null rate of call #1 separately from calls #2..N.
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = "/Users/hsperr/code/youtube_coding/StackIt"
BIN = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/c_engine/stackit"
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
TRIALS = int(sys.argv[3]) if len(sys.argv) > 3 else 10

CELLS = " ".join("1:1" if i in (0, 6, 12, 18, 24) else
                 ("1:2" if i in (4, 8, 16, 20) else "0:0") for i in range(25))
REQ = f"5 5 1 {BUDGET} 40 {CELLS}\n"


def one(_):
    pr = subprocess.Popen([BIN, "--serve"], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True, bufsize=1)
    try:
        out = []
        for _ in range(6):
            pr.stdin.write(REQ); pr.stdin.flush()
            out.append(json.loads(pr.stdout.readline()))
        return out
    finally:
        pr.kill()


print(f"binary={os.path.basename(BIN)} budget={BUDGET}s  load={os.getloadavg()[0]:.1f}")
for k in (1, 4, 10):
    first, later = [], []
    for _ in range(TRIALS):
        with ThreadPoolExecutor(k) as ex:
            for rs in ex.map(one, range(k)):
                first.append(rs[0]); later.extend(rs[1:])
    fn = sum(r["move"] is None for r in first)
    ln = sum(r["move"] is None for r in later)
    print(f"{k:>3} engines spawned together: "
          f"call#1 null {fn:>3}/{len(first):<3} = {fn/len(first):>4.0%}   "
          f"calls#2-6 null {ln:>4}/{len(later):<4} = {ln/len(later):>4.0%}   "
          f"call#1 median secs {sorted(r['secs'] for r in first)[len(first)//2]:.4f}   "
          f"load {os.getloadavg()[0]:.1f}", flush=True)
