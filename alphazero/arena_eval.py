"""In-process evaluation: play the current net against baselines and the prior
best. Everything is the same checkout, so no subprocess is needed (unlike the
top-level arena.py, which exists to load *different* code versions at once).

Players expose `move(board, rng) -> (x, y)`. Colors alternate across games so
first-player advantage cancels out.
"""
import json
import os
import subprocess

import numpy as np

from board import Board
from .mcts_az import MCTS
from .selfplay import game_winner
from .encoding import index_to_move


class AZPlayer:
    """MCTS+net player. Always plays the search's `selected_action` (never an
    argmax/sample of the policy, which could pick a Gumbel-eliminated action).
    Early game variety comes from search noise on the first `explore_plies` plies
    (Gumbel at the root for fixed-sim search, Dirichlet for the wall-clock path),
    which perturbs the selected action without ever playing an eliminated one."""

    def __init__(self, evaluator, cfg, explore_plies=4):
        self.mcts = MCTS(evaluator, cfg)
        self.explore_plies = explore_plies
        self._ply = 0
        # >0 => search by wall-clock (s/move) instead of a fixed sim count
        self.time_budget = getattr(cfg, "az_time_budget", 0.0) or None

    def reset(self):
        self._ply = 0

    def move(self, board, rng):
        explore = self._ply < self.explore_plies         # noise -> early variety
        if self.time_budget:
            counts, root = self.mcts.search(board, add_noise=explore, rng=rng,
                                            time_budget=self.time_budget, max_sims=1_000_000)
        else:
            counts, root = self.mcts.search(board, add_noise=explore, rng=rng)
        if counts.sum() == 0:
            return None
        self._ply += 1
        return index_to_move(root.selected_action, board.size_x)


class CAlphaBetaPlayer:
    """AlphaBeta played by the C engine in `c_engine/` instead of `alphabeta.py`.

    Same algorithm, ~50x the nodes/sec, so at equal wall-clock it searches about
    4 ply deeper. The binary is spoken to over a line protocol on a long-lived
    subprocess (one per worker process, reused across games) — spawning per move
    would cost more than the search at small budgets.

    Request : "<sx> <sy> <side> <secs> <max_depth> <v:o> <v:o> ..." (cells in
              ascending index order, index = y*sx + x)
    Response: one JSON line, {"move": [x, y] | null, ...}
    """

    _shared = {}          # binary path -> Popen, one per worker process

    def __init__(self, budget=0.05, binary=None, max_depth=40):
        self.budget = budget
        self.max_depth = max_depth
        self.binary = binary or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "c_engine", "stackit")

    def reset(self):
        pass

    def _proc(self):
        pr = CAlphaBetaPlayer._shared.get(self.binary)
        if pr is None or pr.poll() is not None:
            pr = subprocess.Popen([self.binary, "--serve"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
            CAlphaBetaPlayer._shared[self.binary] = pr
        return pr

    def move(self, board, rng):
        sx, sy = board.size_x, board.size_y
        cells = " ".join(f"{board.board[y][x]}:{board.player[y][x]}"
                         for y in range(sy) for x in range(sx))
        req = f"{sx} {sy} {board.current_player} {self.budget} {self.max_depth} {cells}\n"
        pr = self._proc()
        pr.stdin.write(req)
        pr.stdin.flush()
        line = pr.stdout.readline()
        if not line:                                  # engine died — do not fail the match
            raise RuntimeError(f"C AlphaBeta engine died ({self.binary})")
        mv = json.loads(line).get("move")
        return tuple(mv) if mv else None


def make_alphabeta(cfg, budget):
    """The fixed-budget AlphaBeta opponent, C or Python per `cfg.alphabeta_engine`.
    Falls back to Python (loudly) if the C binary has not been built."""
    if getattr(cfg, "alphabeta_engine", "python") == "c":
        p = CAlphaBetaPlayer(budget)
        if os.path.exists(p.binary):
            return p
        print(f"[arena] C engine not built at {p.binary} — using the Python AlphaBeta")
    return AlphaBetaPlayer(budget)


class RandomPlayer:
    def reset(self):
        pass

    def move(self, board, rng):
        moves = board.possible_moves()
        return moves[rng.integers(len(moves))] if moves else None


class AlphaBetaPlayer:
    def __init__(self, budget=0.05, max_depth=12):
        from alphabeta import AlphaBeta
        self.engine = AlphaBeta()
        self.budget = budget
        self.max_depth = max_depth

    def reset(self):
        pass

    def move(self, board, rng):
        mv, _ = self.engine.get_best_move(board, thinking_time=self.budget,
                                          max_depth=self.max_depth)
        return mv


def play_match(player_a, player_b, n_games, board_size, max_plies, rng,
               opening_random_plies=0):
    """Return (wins_a, wins_b, draws). Colors swap every game."""
    wa = wb = draws = 0
    for g in range(n_games):
        # even game: A is player1; odd game: B is player1
        p1, p2 = (player_a, player_b) if g % 2 == 0 else (player_b, player_a)
        player_a.reset(); player_b.reset()
        board = Board(board_size, board_size)
        players = {1: p1, 2: p2}
        # Random opening plies: without them two strong nets replay nearly the
        # same game every time, so an N-game match carries far less than N games
        # of information. Self-play has always done this; the arena had not.
        for _ in range(opening_random_plies):
            legal = board.possible_moves()
            if not legal:
                break
            board.move(*legal[rng.integers(len(legal))])
        for _ in range(max_plies):
            if board.winning_player() or not board.possible_moves():
                break
            mv = players[board.current_player].move(board, rng)
            if mv is None or tuple(mv) not in board.possible_moves():
                break
            board.move(*mv)
        w = game_winner(board)                # 0 / 1 / 2
        if w == 0:
            draws += 1
        else:
            winner_player = p1 if w == 1 else p2
            if winner_player is player_a:
                wa += 1
            else:
                wb += 1
    return wa, wb, draws


def win_rate(wins, losses, draws):
    total = wins + losses + draws
    return (wins + 0.5 * draws) / total if total else 0.0
