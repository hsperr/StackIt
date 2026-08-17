"""AlphaZero as a drop-in StackIt engine.

Implements the same `get_best_move(board, thinking_time, max_depth, show_perft)
-> (move, score)` contract as AlphaBeta/MCTS/MinMax, so a trained net plugs
straight into arena.py and server.py. Plays greedily (argmax visit count, no
Dirichlet noise). `score` is the root value estimate for the chosen move, in
[-1, 1], from the side-to-move's perspective.
"""
import os
import copy

import numpy as np
import torch

from .config import Config
from .net import Evaluator
from .mcts_az import MCTS
from .metrics import load_net
from .encoding import action_index, index_to_move


class AlphaZero:
    def __init__(self, ckpt=None, sims=160, device="cpu", max_sims_cap=2000):
        self.cfg = Config()
        self.ckpt = ckpt or os.path.join(self.cfg.ckpt_dir, "best.pt")
        self.device = torch.device(device)
        self.sims = sims
        self.max_sims_cap = max_sims_cap
        self.net = None
        self.mcts = None
        self.perft = []

    def __repr__(self):
        return "AlphaZero()"

    def _ensure_loaded(self, board):
        if self.net is not None:
            return
        if not os.path.exists(self.ckpt):
            raise FileNotFoundError(
                f"No checkpoint at {self.ckpt}. Train first: python3 -m alphazero.train")
        self.net, ckpt = load_net(self.ckpt, self.device)
        trained = ckpt["arch"]["board_size"]
        if trained != board.size_x or board.size_x != board.size_y:
            raise ValueError(
                f"Checkpoint is for {trained}x{trained} boards; got "
                f"{board.size_x}x{board.size_y}. Train a net for this size.")
        ev = Evaluator(self.net, self.device)
        self.mcts = MCTS(ev, self.cfg)

    def get_best_move(self, board, thinking_time=None, max_depth=None, show_perft=False):
        self._ensure_loaded(board)
        self.perft = []

        moves = board.possible_moves()
        if not moves:
            return None, None
        if len(moves) == 1:
            return moves[0], None

        # Use the time budget if given (cap sims high), else a fixed sim count.
        if thinking_time:
            counts, root = self.mcts.search(board, add_noise=False,
                                            time_budget=thinking_time,
                                            max_sims=self.max_sims_cap)
        else:
            counts, root = self.mcts.search(board, add_noise=False, max_sims=self.sims)

        nx = board.size_x
        # Play the action the search selected (Sequential-Halving survivor for the
        # Gumbel path, most-visited child for PUCT) — NOT a raw child_N argmax, whose
        # ties among final Gumbel candidates would pick arbitrarily.
        best_action = root.selected_action
        best_local = int(np.where(root.legal == best_action)[0][0])
        move = index_to_move(best_action, nx)
        n = root.child_N[best_local]
        score = float(root.child_W[best_local] / n) if n > 0 else None

        if show_perft:
            order = np.argsort(root.child_N)[::-1]
            for li in order[:8]:
                a = int(root.legal[li])
                nn = root.child_N[li]
                q = root.child_W[li] / nn if nn > 0 else 0.0
                self.perft.append([index_to_move(a, nx), int(nn), round(float(q), 3),
                                   round(float(root.priors[li]), 3)])
            print(f"AlphaZero sims={int(root.child_N.sum())}")
            for p in self.perft:
                print("  move", p[0], "N", p[1], "Q", p[2], "P", p[3])

        return move, score

    def analyze(self, board, thinking_time=5.0, top_k=6, pv_len=8):
        """Search `board` for up to `thinking_time` seconds and return the best
        move plus the search's own view of the position: a win estimate, sim
        count, the most-visited moves, and the principal variation (the line it
        expects). Used by the dashboard's live 'play the best model' panel."""
        self._ensure_loaded(board)
        moves = board.possible_moves()
        if not moves:
            return None
        # raw net read of this position, BEFORE search (the net's "intuition")
        net_policy, net_value = self.mcts.ev.infer(board)
        _, root = self.mcts.search(board, add_noise=False,
                                   time_budget=thinking_time, max_sims=self.max_sims_cap)
        nx = board.size_x
        best_local = int(np.argmax(root.child_N))
        move = index_to_move(int(root.legal[best_local]), nx)
        n = root.child_N[best_local]
        q = float(root.child_W[best_local] / n) if n > 0 else 0.0
        order = np.argsort(root.child_N)[::-1][:top_k]
        top = [{"move": list(index_to_move(int(root.legal[i]), nx)),
                "visits": int(root.child_N[i]),
                "q": round(float(root.child_W[i] / root.child_N[i]) if root.child_N[i] > 0 else 0.0, 3),
                "prior": round(float(root.priors[i]), 3)} for i in order]
        return {"move": list(move), "q": round(q, 3),
                "win_prob": round((q + 1) / 2, 3),           # search win estimate
                "sims": int(root.child_N.sum()),
                "top": top,
                "pv": [list(m) for m in principal_variation(root, nx, pv_len)],
                # raw net heads for the same position (no search)
                "net_value": round(float(net_value), 3),
                "net_win_prob": round((float(net_value) + 1) / 2, 3),
                "net_policy": [round(float(p), 4) for p in net_policy]}


def principal_variation(root, size_x, max_len=8):
    """Follow the most-visited child from the root down the tree — the line the
    search currently believes is best."""
    pv, node = [], root
    for _ in range(max_len):
        if node is None or not node.expanded or node.is_terminal or node.child_N is None:
            break
        i = int(np.argmax(node.child_N))
        if node.child_N[i] == 0:
            break
        pv.append(index_to_move(int(node.legal[i]), size_x))
        node = node.children.get(i)
    return pv
