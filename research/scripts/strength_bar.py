"""A strength bar that gives the same answer twice.

WHY THE OLD ONE LIED
--------------------
1. The C AlphaBeta played to a WALL-CLOCK budget (0.05 s/move), so its strength
   was a function of how busy the machine was, and search_best_move() commits
   result.best_move only if a depth finishes inside that budget -- otherwise it
   answers {"move":null}.
2. alphazero/parallel._match_chunk turns a null (or illegal) move into `break`,
   and alphazero.selfplay.game_winner then scores the HALF-PLAYED position by box
   count. After 3 random opening plies player 1 is up 2 chips to 1, so an engine
   that fails to move hands the point to its opponent. Eight games of that score
   4W-0L-4D = 0.75 for the net, in 0.3 seconds, which is exactly what the trainer
   logged at iterations 12/24/33/36/39/42.

WHAT THIS DOES INSTEAD
----------------------
* AlphaBeta plays to a FIXED DEPTH: request secs=1e9, max_depth=D. The wall clock
  never fires, iterative deepening always completes depth D, best_move is always
  committed. Verified identical at 1 and 10 concurrent engines
  (research/scripts/ab_fixeddepth_check.py). Strength no longer depends on load.
* An abort is a LOUD FAILURE, never a result: any None / illegal move is counted,
  reported, and excluded from the score. If aborts > 0 the number is not
  trustworthy and the script says so.
* The net side is fixed-sim MCTS with a per-game seeded rng, so the whole match
  is deterministic; repeating a seed block must reproduce the scoreline exactly.
* Records the load average, the engine binary and its mtime, so no two numbers
  can ever be silently compared across different conditions.

Usage:
    python research/scripts/strength_bar.py <ckpt> [games] [sims] [depth] [workers] [blocks]

`blocks` (default 3) runs that many DISJOINT seed blocks, then repeats block 0 to
prove determinism. Spread across blocks is the real sampling variance.
"""
import json, os, sys, time
from multiprocessing import get_context

ROOT = "/Users/hsperr/code/youtube_coding/StackIt"
sys.path.insert(0, ROOT)
_CTX = get_context("spawn")

HUGE_SECS = 1e9            # so the engine's wall clock never fires
DEFAULT_BIN = f"{ROOT}/c_engine/stackit"


def _init():
    import torch
    torch.set_num_threads(1)


class FixedDepthAB:
    """The C engine at a fixed depth. Same line protocol as
    alphazero.arena_eval.CAlphaBetaPlayer, but max_depth is the limit that binds
    and the time limit is effectively infinite."""

    _shared = {}

    def __init__(self, binary, depth):
        self.binary, self.depth = binary, depth

    def reset(self):
        pass

    def _proc(self):
        import subprocess
        pr = FixedDepthAB._shared.get(self.binary)
        if pr is None or pr.poll() is not None:
            pr = subprocess.Popen([self.binary, "--serve"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
            FixedDepthAB._shared[self.binary] = pr
        return pr

    def move(self, board, rng):
        cells = " ".join(f"{board.board[y][x]}:{board.player[y][x]}"
                         for y in range(board.size_y) for x in range(board.size_x))
        pr = self._proc()
        pr.stdin.write(f"{board.size_x} {board.size_y} {board.current_player} "
                       f"{HUGE_SECS} {self.depth} {cells}\n")
        pr.stdin.flush()
        line = pr.stdout.readline()
        if not line:
            raise RuntimeError(f"C AlphaBeta engine died ({self.binary})")
        mv = json.loads(line).get("move")
        return tuple(mv) if mv else None


def _chunk(payload):
    import numpy as np, torch
    torch.set_num_threads(1)
    from board import Board
    from alphazero.config import Config
    from alphazero.net import build_net, Evaluator
    from alphazero.selfplay import game_winner
    from alphazero.arena_eval import AZPlayer

    arch, state, binary, depth, specs, cfg_dict = payload
    cfg = Config(**cfg_dict)
    net = build_net(arch, state)
    net.load_state_dict(state)
    net.eval()
    ev = Evaluator(net, torch.device("cpu"))

    out = []
    for net_is_p1, seed in specs:
        rng = np.random.default_rng(int(seed))
        pa, pb = AZPlayer(ev, cfg), FixedDepthAB(binary, depth)
        pa.reset(); pb.reset()
        p1, p2 = (pa, pb) if net_is_p1 else (pb, pa)
        players = {1: p1, 2: p2}
        b = Board(cfg.board_size, cfg.board_size)
        for _ in range(cfg.match_random_plies):
            legal = b.possible_moves()
            if not legal:
                break
            b.move(*legal[rng.integers(len(legal))])
        plies, abort = 0, None
        for _ in range(cfg.max_game_plies):
            if b.winning_player() or not b.possible_moves():
                break
            who = players[b.current_player]
            mv = who.move(b, rng)
            if mv is None or tuple(mv) not in b.possible_moves():
                # NEVER scored. A player that cannot move is a broken measurement.
                abort = ("net" if who is pa else "ab") + ("/none" if mv is None else "/illegal")
                break
            b.move(*mv)
            plies += 1
        if abort:
            out.append({"seed": int(seed), "res": "ABORT", "why": abort, "plies": plies})
            continue
        w = game_winner(b)
        res = "draw" if w == 0 else ("win" if (p1 if w == 1 else p2) is pa else "loss")
        out.append({"seed": int(seed), "res": res, "plies": plies})
    return out


def _split(items, n):
    return [c for c in (items[i::n] for i in range(n)) if c]


def play(pool, arch, state, binary, depth, cfg, games, base_seed, workers):
    specs = [(g % 2 == 0, base_seed + g) for g in range(games)]
    payloads = [(arch, state, binary, depth, c, cfg.to_dict()) for c in _split(specs, workers)]
    return [r for res in pool.map(_chunk, payloads) for r in res]


def score(rows):
    w = sum(r["res"] == "win" for r in rows)
    l = sum(r["res"] == "loss" for r in rows)
    d = sum(r["res"] == "draw" for r in rows)
    a = sum(r["res"] == "ABORT" for r in rows)
    n = w + l + d
    return w, l, d, a, ((w + 0.5 * d) / n if n else float("nan"))


def main():
    from dataclasses import replace
    from alphazero.config import Config
    from alphazero.metrics import load_net
    from alphazero.parallel import net_arch_state

    ckpt = sys.argv[1]
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    sims = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    depth = int(sys.argv[4]) if len(sys.argv) > 4 else 7
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 6
    blocks = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    binary = sys.argv[7] if len(sys.argv) > 7 else DEFAULT_BIN

    path = ckpt if ckpt.startswith("/") else os.path.join(ROOT, ckpt)
    net, meta = load_net(path, "cpu")
    arch, state = net_arch_state(net)
    cfg = replace(Config(match_random_plies=3), num_simulations=sims,
                  eval_simulations=sims)
    print(f"ckpt   {ckpt}  (iter {meta.get('extra', {}).get('iter')})")
    print(f"net    {arch}  sims={sims}")
    print(f"AB     {os.path.basename(binary)} @ FIXED DEPTH {depth} "
          f"(mtime {time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(binary)))})")
    print(f"host   load {os.getloadavg()[0]:.1f}  workers={workers}  "
          f"{games} games/block x {blocks} blocks + 1 determinism replay")
    print()

    pool = _CTX.Pool(workers, initializer=_init)
    try:
        results = []
        for bl in range(blocks):
            t = time.time()
            rows = play(pool, arch, state, binary, depth, cfg, games,
                        7_000_000 + bl * 100_000, workers)
            w, l, d, a, wr = score(rows)
            results.append((f"block {bl}", w, l, d, a, wr, time.time() - t,
                            os.getloadavg()[0]))
            print(f"block {bl}: {w:>3}W-{l:>3}L-{d:>2}D  aborts={a}  "
                  f"winrate {wr:.3f}   ({time.time()-t:.0f}s, load {os.getloadavg()[0]:.1f})",
                  flush=True)
        t = time.time()
        rows = play(pool, arch, state, binary, depth, cfg, games, 7_000_000, workers)
        w, l, d, a, wr = score(rows)
        print(f"replay of block 0: {w:>3}W-{l:>3}L-{d:>2}D  aborts={a}  winrate {wr:.3f}"
              f"   ({time.time()-t:.0f}s, load {os.getloadavg()[0]:.1f})", flush=True)
        same = (w, l, d) == results[0][1:4]
        print(f"deterministic: {'YES (identical scoreline)' if same else 'NO'}")
        wrs = [r[5] for r in results]
        mean = sum(wrs) / len(wrs)
        spread = max(wrs) - min(wrs)
        se = (mean * (1 - mean) / (games * len(results))) ** 0.5
        print(f"\n{ckpt}: winrate {mean:.3f} over {games*len(results)} games "
              f"(blocks {['%.3f' % x for x in wrs]}, spread {spread:.3f}, "
              f"binomial se {se:.3f} -> +-{2*se:.3f} at 2 se)")
        total_ab = sum(r[4] for r in results)
        if total_ab:
            print(f"WARNING: {total_ab} aborted games -- number NOT trustworthy")
    finally:
        pool.close(); pool.join()


if __name__ == "__main__":
    main()
