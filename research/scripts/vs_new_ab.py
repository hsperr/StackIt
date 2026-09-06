"""The live net vs the OLD and the NEW AlphaBeta eval, same net, same budget.

Why not reuse alphazero.parallel.match_parallel: it builds the opponent through
make_alphabeta(), which hard-codes c_engine/stackit. We need to point the same
match at a second binary (c_engine/stackit_new, EVAL_VARIANT=4) without touching
anything under alphazero/ while a training run is live.

The __main__ guard is load-bearing (spawn pool re-imports this file).
"""
import sys, time
from multiprocessing import get_context

import numpy as np
import torch

ROOT = '/Users/hsperr/code/youtube_coding/StackIt'
sys.path.insert(0, ROOT)

from board import Board
from alphazero.config import Config
from alphazero.metrics import load_net
from alphazero.net import StackNet, Evaluator
from alphazero.parallel import net_arch_state
from alphazero.selfplay import game_winner
from alphazero.arena_eval import AZPlayer, CAlphaBetaPlayer, win_rate

_CTX = get_context("spawn")


def _init():
    torch.set_num_threads(1)


def _chunk(payload):
    torch.set_num_threads(1)
    arch, state, binary, budget, specs, cfg_dict = payload
    cfg = Config(**cfg_dict)
    net = StackNet(arch["board_size"], arch["channels"], arch["res_blocks"])
    net.load_state_dict(state)
    net.eval()
    ev = Evaluator(net, torch.device("cpu"))

    out = []
    for first_is_a, seed in specs:
        rng = np.random.default_rng(int(seed))
        pa = AZPlayer(ev, cfg)
        pb = CAlphaBetaPlayer(budget, binary=binary)
        pa.reset(); pb.reset()
        p1, p2 = (pa, pb) if first_is_a else (pb, pa)
        players = {1: p1, 2: p2}
        board = Board(cfg.board_size, cfg.board_size)
        for _ in range(getattr(cfg, "match_random_plies", 0)):
            legal = board.possible_moves()
            if not legal:
                break
            board.move(*legal[rng.integers(len(legal))])
        for _ in range(cfg.max_game_plies):
            if board.winning_player() or not board.possible_moves():
                break
            mv = players[board.current_player].move(board, rng)
            if mv is None or tuple(mv) not in board.possible_moves():
                break
            board.move(*mv)
        w = game_winner(board)
        if w == 0:
            out.append("draw")
        else:
            out.append("a" if (p1 if w == 1 else p2) is pa else "b")
    return out


def _split(items, n):
    return [c for c in (items[i::n] for i in range(n)) if c]


def match(arch, state, binary, budget, games, seed, pool, workers, cfg):
    specs = [(g % 2 == 0, seed + g) for g in range(games)]
    payloads = [(arch, state, binary, budget, c, cfg.to_dict())
                for c in _split(specs, workers)]
    wa = wb = dr = 0
    for res in pool.map(_chunk, payloads):
        for r in res:
            wa += r == "a"; wb += r == "b"; dr += r == "draw"
    return wa, wb, dr


def main():
    net_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_cover/best.pt"
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    sims = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    cfg = Config(num_simulations=sims, eval_simulations=sims,
                 match_random_plies=3, alphabeta_engine="c")
    net, meta = load_net(f"{ROOT}/{net_path}" if not net_path.startswith("/") else net_path, "cpu")
    arch, state = net_arch_state(net)
    print(f"net={net_path} v={meta.get('extra', {})} sims={sims} "
          f"games={games} workers={workers}", flush=True)

    ENGINES = (("old chips-only", f"{ROOT}/c_engine/stackit"),
               ("new territory+threat", f"{ROOT}/c_engine/stackit_new"))
    pool = _CTX.Pool(workers, initializer=_init)
    try:
        for budget in (0.05, 0.3):
            for label, binary in ENGINES:
                t = time.time()
                wa, wb, dr = match(arch, state, binary, budget, games, 4242,
                                   pool, workers, cfg)
                print(f"  {budget}s/move vs {label:22s}: "
                      f"{wa}W-{wb}L-{dr}D = {win_rate(wa, wb, dr):.0%}  "
                      f"({time.time() - t:.0f}s)", flush=True)
    finally:
        pool.close(); pool.join()


if __name__ == "__main__":
    main()
