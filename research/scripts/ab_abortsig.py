"""What does an aborted AlphaBeta match score, and how long does it take?

_match_chunk `break`s when a player returns None or an illegal move, and
game_winner() then scores the half-played position by box count. After the 3
random opening plies player 1 has 2 chips and player 2 has 1, and it is player
2's turn -- so whichever side fails, the OTHER side is handed the point.

Runs the real match loop single-process with one side stubbed out, so the
scoreline and the wall clock of a fully-aborted 8-game match can be compared
against the trainer's book.
"""
import sys, time
sys.path.insert(0, "/Users/hsperr/code/youtube_coding/StackIt")
from dataclasses import replace

import numpy as np, torch

from board import Board
from alphazero.config import Config
from alphazero.metrics import load_net
from alphazero.net import Evaluator
from alphazero.selfplay import game_winner
from alphazero.arena_eval import AZPlayer, CAlphaBetaPlayer, win_rate

torch.set_num_threads(2)


class Dead:
    """A player that never returns a move (what the C engine does when it
    answers {"move":null}, or what any desynced/illegal reply amounts to)."""
    def reset(self): pass
    def move(self, board, rng): return None


def run(label, kill, games=8, seed=3973000):
    cfg = replace(Config(match_random_plies=3, alphabeta_engine="c"), num_simulations=128)
    net, _ = load_net("checkpoints_cover/versions/v39.pt", "cpu")
    ev = Evaluator(net, torch.device("cpu"))
    wa = wb = dr = 0
    t = time.time()
    for g in range(games):
        rng = np.random.default_rng(seed + g)
        pa = Dead() if kill == "net" else AZPlayer(ev, cfg)
        pb = Dead() if kill == "ab" else CAlphaBetaPlayer(0.05)
        pa.reset(); pb.reset()
        p1, p2 = (pa, pb) if g % 2 == 0 else (pb, pa)
        players = {1: p1, 2: p2}
        b = Board(cfg.board_size, cfg.board_size)
        for _ in range(cfg.match_random_plies):
            legal = b.possible_moves()
            if not legal: break
            b.move(*legal[rng.integers(len(legal))])
        for _ in range(cfg.max_game_plies):
            if b.winning_player() or not b.possible_moves(): break
            mv = players[b.current_player].move(b, rng)
            if mv is None or tuple(mv) not in b.possible_moves(): break
            b.move(*mv)
        w = game_winner(b)
        if w == 0: dr += 1
        elif (p1 if w == 1 else p2) is pa: wa += 1
        else: wb += 1
    print(f"{label:34s} net {wa}W-{wb}L-{dr}D = {win_rate(wa, wb, dr):.3f}   "
          f"{time.time()-t:.2f}s", flush=True)


if __name__ == "__main__":
    print("trainer book, v24 and v39      net 4W-0L-4D = 0.750   eval block 0.3s")
    print("trainer book, v36 and v42      net 4W-1L-3D = 0.688   eval block 0.3s")
    print("trainer book, v12              net 4W-3L-1D = 0.562   eval block 0.3s")
    run("AlphaBeta never returns a move", "ab")
    run("the net never returns a move",   "net")
