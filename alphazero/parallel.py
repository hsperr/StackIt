"""Play independent games across CPU cores with a process pool.

Self-play and evaluation are both just "play N independent games", and on this
machine that's ~85% of an iteration's wall-clock. MCTS is single-threaded Python
(GIL-bound), so real parallelism means separate processes. Each worker rebuilds
the net on CPU with 1 torch thread (so 8 workers don't oversubscribe cores) and
plays a slice of the games.

The pool is created once and reused; each call ships the current net weights
(~1MB) to the workers.
"""
from dataclasses import replace
from multiprocessing import get_context

import numpy as np
import torch

from board import Board
from .config import Config
from .net import StackNet, build_net, Evaluator
from .selfplay import play_game, game_winner
from .arena_eval import AZPlayer, RandomPlayer, make_alphabeta

_CTX = get_context("spawn")
_CPU = torch.device("cpu")


def _init_worker():
    torch.set_num_threads(1)


def _build_net(arch, state):
    net = build_net(arch, state)
    net.load_state_dict(state)
    net.eval()
    return net


def _cpu_state(net):
    return {k: v.detach().cpu() for k, v in net.state_dict().items()}


def make_pool(n_workers):
    return _CTX.Pool(n_workers, initializer=_init_worker)


def _split(items, n):
    """Round-robin split into <= n non-empty chunks (balances uneven game lengths)."""
    chunks = [items[i::n] for i in range(n)]
    return [c for c in chunks if c]


# ------------------------------------------------------------------ self-play
def _sp_chunk(payload):
    torch.set_num_threads(1)
    arch, state, cfg_dict, seeds, opp_pool = payload
    ev = Evaluator(_build_net(arch, state), _CPU)
    opp_evs = [Evaluator(_build_net(oa, os_), _CPU) for (oa, os_) in opp_pool]
    base = Config(**cfg_dict)
    examples, last = [], None
    for s in seeds:
        cfg = replace(base, seed=int(s))
        rng = np.random.default_rng(int(s))
        # a fraction of games are vs a pool opponent (diversity + robustness)
        opp = None
        if opp_evs and rng.random() < cfg.pool_play_frac:
            opp = opp_evs[rng.integers(len(opp_evs))]
        ex, rec = play_game(ev, cfg, rng, opponent_ev=opp)
        examples.extend(ex)
        if rec and opp is None:          # dashboard replay: prefer a pure self-play game
            last = rec
        elif last is None:
            last = rec
    return examples, last


def selfplay_parallel(net, cfg, n_games, base_seed, pool, n_workers, opponent_pool=None):
    """Returns (all_examples, last_record). Examples are raw (planes, pi, z);
    the caller applies symmetry augmentation. `opponent_pool` is a list of
    (arch, state) mixed in as self-play opponents for a fraction of games."""
    arch, state = net.arch(), _cpu_state(net)
    opp_pool = opponent_pool or []
    seeds = [base_seed + i for i in range(n_games)]
    payloads = [(arch, state, cfg.to_dict(), c, opp_pool) for c in _split(seeds, n_workers)]
    examples, last = [], None
    for ex, rec in pool.map(_sp_chunk, payloads):
        examples.extend(ex)
        if rec:
            last = rec
    return examples, last


# ---------------------------------------------------------------------- matches
def _match_chunk(payload):
    torch.set_num_threads(1)
    arch, a_state, opp, specs, board_size, max_plies, cfg_dict = payload
    cfg = Config(**cfg_dict)
    a_ev = Evaluator(_build_net(arch, a_state), _CPU)
    if opp[0] == "az":                              # ("az", opp_arch, opp_state)
        b_ev = Evaluator(_build_net(opp[1], opp[2]), _CPU)

    def make_a():
        return AZPlayer(a_ev, cfg)

    def make_b():
        if opp[0] == "az":
            return AZPlayer(b_ev, cfg)
        if opp[0] == "random":
            return RandomPlayer()
        if opp[0] == "alphabeta":
            return make_alphabeta(cfg, opp[1])
        # Anything else is a caller bug, and it used to fall through to the line
        # above: research/scripts/head2head.py passed a bare (arch, state) pair,
        # so `state` was handed to the C engine as its seconds-per-move, the
        # protocol line was garbage, the engine answered null every time, and the
        # "net A vs net B" number it printed was really net-A-versus-nothing.
        raise ValueError(
            f"unknown opponent spec {opp[0]!r}; expected ('az', arch, state), "
            f"('random',) or ('alphabeta', budget)")

    out = []
    for first_is_a, seed in specs:
        rng = np.random.default_rng(int(seed))
        pa, pb = make_a(), make_b()
        pa.reset(); pb.reset()
        p1, p2 = (pa, pb) if first_is_a else (pb, pa)
        players = {1: p1, 2: p2}
        board = Board(board_size, board_size)
        # Random opening plies, same reason as in arena_eval.play_match: without
        # them two strong players replay nearly the same game every time and an
        # N-game match is worth far less than N games.
        for _ in range(getattr(cfg, "match_random_plies", 0)):
            legal = board.possible_moves()
            if not legal:
                break
            board.move(*legal[rng.integers(len(legal))])
        for _ in range(max_plies):
            if board.winning_player() or not board.possible_moves():
                break
            mv = players[board.current_player].move(board, rng)
            if mv is None or tuple(mv) not in board.possible_moves():
                # NEVER score an aborted game. This used to `break`, and
                # game_winner() then scored the half-played board by box count --
                # after the random opening plies player 1 leads 2 chips to 1, so a
                # player that failed to move HANDED the point to its opponent.
                # That is the entire "the net beats AlphaBeta 75%" result: six
                # iterations whose 8-game match finished in 0.3s with exactly 4
                # net wins, one per game where AlphaBeta moved first. An abort is
                # a bug in a player, so it has to be loud.
                from .mcts_az import terminal_value
                raise RuntimeError(
                    f"match aborted: player {board.current_player} returned "
                    f"{mv!r} with {len(board.possible_moves())} legal moves "
                    f"available; winning_player={board.winning_player()} "
                    f"terminal_value={terminal_value(board)} "
                    f"board={[list(r) for r in board.board]} "
                    f"owner={[list(r) for r in board.player]}")
            board.move(*mv)
        w = game_winner(board)
        if w == 0:
            out.append("draw")
        else:
            winner = p1 if w == 1 else p2
            out.append("a" if winner is pa else "b")
    return out


def net_arch_state(net):
    return net.arch(), _cpu_state(net)


def match_parallel(a_arch, a_state, opp, cfg, n_games, base_seed, pool, n_workers):
    """Play side A (a_arch/a_state) vs an opponent over n_games, colors
    alternating. `opp` is ("az", opp_arch, opp_state) | ("random", None) |
    ("alphabeta", budget). Returns (wins_a, wins_b, draws)."""
    specs = [(g % 2 == 0, base_seed + g) for g in range(n_games)]
    payloads = [(a_arch, a_state, opp, c, cfg.board_size, cfg.max_game_plies, cfg.to_dict())
                for c in _split(specs, n_workers)]
    wa = wb = dr = 0
    for res in pool.map(_match_chunk, payloads):
        for r in res:
            if r == "a":
                wa += 1
            elif r == "b":
                wb += 1
            else:
                dr += 1
    return wa, wb, dr
