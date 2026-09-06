"""Does the C engine refuse to move in the positions the gauntlet actually starts from?

Every 0.3 s iteration scored EXACTLY 4 net wins = the 4 even-numbered games, the
ones where AlphaBeta is first to move after the 3 random opening plies. So the
suspect is AlphaBeta failing on that specific ply-3 position. Rebuilds each
iteration's 8 openings from its seed (base = cfg.seed + it*100000 + 73000) and
asks the engine directly.
"""
import json, subprocess, sys
import numpy as np
sys.path.insert(0, "/Users/hsperr/code/youtube_coding/StackIt")
from board import Board

BIN = "/Users/hsperr/code/youtube_coding/StackIt/c_engine/stackit"
pr = subprocess.Popen([BIN, "--serve"], stdin=subprocess.PIPE,
                      stdout=subprocess.PIPE, text=True, bufsize=1)

BAD = {12, 24, 33, 36, 39, 42}      # eval block took 0.3 s, net "won" 4/8
OK = {6, 9, 18, 21, 27}             # eval block took 4-7 s, net scored ~0

for it in sorted(BAD | OK):
    seed = it * 100000 + 73000
    nulls = legal_mismatch = 0
    for g in range(8):
        rng = np.random.default_rng(seed + g)
        b = Board(5, 5)
        for _ in range(3):
            mv = b.possible_moves()
            b.move(*mv[rng.integers(len(mv))])
        cells = " ".join(f"{b.board[y][x]}:{b.player[y][x]}" for y in range(5) for x in range(5))
        pr.stdin.write(f"5 5 {b.current_player} 0.05 40 {cells}\n"); pr.stdin.flush()
        r = json.loads(pr.stdout.readline())
        if r.get("move") is None:
            nulls += 1
        elif tuple(r["move"]) not in b.possible_moves():
            legal_mismatch += 1
    tag = "0.3s/4W" if it in BAD else "real"
    print(f"it={it:>3} ({tag:>7}): nulls {nulls}/8  illegal {legal_mismatch}/8", flush=True)
pr.kill()
