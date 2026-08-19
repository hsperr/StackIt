"""Generate supervised training data from AlphaBeta self-play.

Point of the exercise: self-play RL has plateaued and the AlphaBeta bar turned
out to be unreliable, so we cannot tell whether the *net* can even represent
AlphaBeta-strength play. Imitating AlphaBeta is a clean, non-moving target: if
supervised top-1 also stalls near 0.47, the bottleneck is the architecture
(cf. the 52% self-disagreement under board rotation), not the RL loop.

Teacher: c_engine/stackit_new (EVAL_VARIANT=4, the territory+threat eval).
Targets: pi = one-hot on AlphaBeta's chosen move, z = the game result, own =
final ownership, i.e. exactly the three heads alphazero/net.py already has.

Diversity: without it two deterministic engines replay one game forever. We use
a random opening of 2-8 plies plus an epsilon-random move at any ply; the
recorded target is always AlphaBeta's choice for that position, even on the
plies where the random move is the one actually played.

The __main__ guard is load-bearing (spawn pool re-imports this file).
"""
import sys, time, os
sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')
import numpy as np
from multiprocessing import get_context

from board import Board
from alphazero.encoding import encode, action_index
from alphazero.selfplay import game_winner, _ownership
from alphazero.arena_eval import CAlphaBetaPlayer

BIN = '/Users/hsperr/code/youtube_coding/StackIt/c_engine/stackit_new'
N = 5
EPS = 0.08
MAX_PLIES = 200
_CTX = get_context("spawn")


def _chunk(payload):
    n_games, seed0, budget = payload
    ab = CAlphaBetaPlayer(budget, binary=BIN)
    P, PI, Z, OWN = [], [], [], []
    for g in range(n_games):
        rng = np.random.default_rng(seed0 + g)
        b = Board(N, N)
        for _ in range(int(rng.integers(2, 9))):          # random opening
            legal = b.possible_moves()
            if not legal:
                break
            b.move(*legal[rng.integers(len(legal))])
        pos = []
        for _ in range(MAX_PLIES):
            if b.winning_player() or not b.possible_moves():
                break
            mover = b.current_player
            best = ab.move(b, rng)
            if best is None:
                break
            pi = np.zeros(N * N, dtype=np.float32)
            pi[action_index(best[0], best[1], N)] = 1.0
            pos.append((encode(b), pi, mover))
            play = best
            if rng.random() < EPS:                         # exploration ply
                legal = b.possible_moves()
                play = tuple(legal[rng.integers(len(legal))])
            b.move(*play)
        w = game_winner(b)
        for planes, pi, mover in pos:
            P.append(planes); PI.append(pi)
            Z.append(np.float32(0.0 if w == 0 else (1.0 if mover == w else -1.0)))
            OWN.append(_ownership(b, mover, N))
    return (np.asarray(P, dtype=np.float32), np.asarray(PI, dtype=np.float32),
            np.asarray(Z, dtype=np.float32), np.asarray(OWN, dtype=np.float32))


def main():
    games, workers, budget = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    out = sys.argv[4]
    # Seed offset keeps shards disjoint: without it every shard replays the same
    # games and the extra data is worth nothing.
    seed0 = int(sys.argv[5]) if len(sys.argv) > 5 else 900000
    per = [games // workers + (1 if i < games % workers else 0) for i in range(workers)]
    payloads = [(n, seed0 + i * 100000, budget) for i, n in enumerate(per) if n]
    t = time.time()
    with _CTX.Pool(workers) as pool:
        parts = pool.map(_chunk, payloads)
    P = np.concatenate([p[0] for p in parts]); PI = np.concatenate([p[1] for p in parts])
    Z = np.concatenate([p[2] for p in parts]); OWN = np.concatenate([p[3] for p in parts])
    np.savez_compressed(out, planes=P, pi=PI, z=Z, own=OWN)
    print(f"{games} games -> {len(P)} positions, {len(P)/games:.1f} per game, "
          f"draws z==0: {(Z == 0).mean():.1%}, {time.time()-t:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
