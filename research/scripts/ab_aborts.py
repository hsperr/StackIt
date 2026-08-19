"""Why does the training gauntlet see draws when a fresh match never does?

_match_chunk ends a game with `break` when a player returns None or an illegal
move, then scores the position as it stands. An aborted game therefore scores as
a win/draw for whoever happens to be ahead. This replays the same loop with a
counter on every abort, over enough games to stress the long-lived engine
subprocess the training pool reuses for hours.
"""
import sys
sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')
import numpy as np, torch
from dataclasses import replace
from board import Board
from alphazero.config import Config
from alphazero.metrics import load_net
from alphazero.net import Evaluator
from alphazero.selfplay import game_winner
from alphazero.arena_eval import AZPlayer, CAlphaBetaPlayer

torch.set_num_threads(2)
games = int(sys.argv[1])
cfg = replace(Config(match_random_plies=3, alphabeta_engine="c"), num_simulations=128)
net, _ = load_net("checkpoints_cover/versions/v39.pt", "cpu")
ev = Evaluator(net, torch.device("cpu"))

ab_abort = az_abort = wa = wb = dr = 0
for g in range(games):
    rng = np.random.default_rng(3973000 + g)
    pa, pb = AZPlayer(ev, cfg), CAlphaBetaPlayer(0.05)   # pb reuses the shared Popen
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
        who = players[b.current_player]
        mv = who.move(b, rng)
        if mv is None or tuple(mv) not in b.possible_moves():
            if who is pb: ab_abort += 1
            else: az_abort += 1
            break
        b.move(*mv)
    w = game_winner(b)
    if w == 0: dr += 1
    elif (p1 if w == 1 else p2) is pa: wa += 1
    else: wb += 1
print(f"{games} games: net {wa}W-{wb}L-{dr}D   "
      f"aborts: alphabeta={ab_abort} net={az_abort}")
