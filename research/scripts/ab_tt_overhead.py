"""How much of AlphaBeta's wall-clock budget is spent NOT searching?

search_best_move() takes t0, then malloc()s and memset()s a 100 MB
transposition table (TT_BITS=22 x 24-byte entries) -- all of it inside the timed
budget. Only after that does iterative deepening start, and result.best_move is
committed ONLY if a depth finishes before the clock runs out. So if the memset
plus one depth-0 pass overruns the budget, the engine answers {"move":null} and
the arena scores the game as an abort.

Probing with a near-zero budget makes the fixed overhead directly visible: the
reported "secs" is then almost entirely table clearing.
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = "/Users/hsperr/code/youtube_coding/StackIt"
BIN = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/c_engine/stackit"
CELLS = " ".join("1:1" if i in (0, 6, 12, 18, 24) else
                 ("1:2" if i in (4, 8, 16, 20) else "0:0") for i in range(25))


def probe(budget, n):
    def one(_):
        pr = subprocess.Popen([BIN, "--serve"], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True, bufsize=1)
        try:
            req = f"5 5 1 {budget} 40 {CELLS}\n"
            rs = []
            for _ in range(20):
                pr.stdin.write(req); pr.stdin.flush()
                rs.append(json.loads(pr.stdout.readline()))
            return rs
        finally:
            pr.kill()
    with ThreadPoolExecutor(n) as ex:
        return [r for c in ex.map(one, range(n)) for r in c]


print(f"binary={os.path.basename(BIN)}  load={os.getloadavg()[0]:.1f}")
print("budget  K   median secs   p95 secs   median depth   null%")
for budget in (0.0001, 0.05):
    for k in (1, 4, 10, 16):
        rs = probe(budget, k)
        s = sorted(r["secs"] for r in rs)
        d = sorted(r["depth"] for r in rs)
        nulls = sum(r["move"] is None for r in rs)
        print(f"{budget:<7} {k:<3} {s[len(s)//2]:>11.4f} {s[int(len(s)*.95)]:>10.4f} "
              f"{d[len(d)//2]:>13} {nulls/len(rs):>7.0%}", flush=True)
