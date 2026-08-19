"""How much does machine load cost the wall-clock-budgeted AlphaBeta?

Sends the SAME position to K concurrent engine processes at 0.05s/move and
prints the depth and node count each one reached. If depth falls as K rises,
every winrate_vs_ab_light number is a function of what else the machine was
doing, not of net strength.
"""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

BIN = "/Users/hsperr/code/youtube_coding/StackIt/c_engine/stackit"
# 5x5, side 1, 0.05s, depth 40, a mid-game position (a few chips placed)
CELLS = " ".join("1:1" if i in (0, 6, 12, 18, 24) else
                 ("1:2" if i in (4, 8, 16, 20) else "0:0") for i in range(25))
REQ = f"5 5 1 0.05 40 {CELLS}\n"


def one(_):
    pr = subprocess.Popen([BIN, "--serve"], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True, bufsize=1)
    try:
        out = []
        for _ in range(20):                       # 20 searches, report the median
            pr.stdin.write(REQ); pr.stdin.flush()
            out.append(json.loads(pr.stdout.readline()))
        out.sort(key=lambda r: r["nodes"])
        return out[len(out) // 2]
    finally:
        pr.kill()


for k in (1, 2, 5, 10):
    with ThreadPoolExecutor(k) as ex:
        rs = list(ex.map(one, range(k)))
    d = sorted(r["depth"] for r in rs)
    n = sorted(r["nodes"] for r in rs)
    print(f"{k:>2} concurrent engines: median depth {d[len(d)//2]:>2}  "
          f"median nodes {n[len(n)//2]:>8,}", flush=True)
