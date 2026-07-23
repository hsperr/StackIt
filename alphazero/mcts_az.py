"""PUCT Monte-Carlo Tree Search guided by the neural net.

Unlike the classic MCTS in ../mcts.py (random rollouts to a terminal), here a
newly expanded leaf is evaluated *once* by the net: its policy seeds the child
priors and its value is backed up directly — no rollout. Selection uses PUCT:

    U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))

Values are stored from each parent's own perspective, and the leaf value's sign
is flipped at every ply on the way back up (players alternate).
"""
import time

import numpy as np

from .encoding import action_index, index_to_move


def terminal_value(board):
    """Game-over value from board.current_player's perspective, or None if the
    game is still going. Mirrors the arena's rules: domination, else box count
    when no legal moves remain."""
    w = board.winning_player()
    if w:
        return 1.0 if w == board.current_player else -1.0
    if not board.possible_moves():
        me = board.boxes_for(board.current_player)
        op = board.boxes_for(board.other_player)
        return 1.0 if me > op else (-1.0 if me < op else 0.0)
    return None


class Node:
    __slots__ = ("board", "to_play", "expanded", "is_terminal", "terminal_value",
                 "legal", "priors", "child_N", "child_W", "children")

    def __init__(self, board):
        self.board = board
        self.to_play = board.current_player
        self.expanded = False
        self.is_terminal = False
        self.terminal_value = 0.0
        self.legal = None
        self.priors = None
        self.child_N = None
        self.child_W = None
        self.children = None

    def expand(self, evaluator):
        tv = terminal_value(self.board)
        if tv is not None:
            self.is_terminal = True
            self.terminal_value = tv
            self.expanded = True
            return tv
        priors_full, v = evaluator.infer(self.board)
        nx = self.board.size_x
        legal = np.array([action_index(x, y, nx)
                          for (x, y) in self.board.possible_moves()], dtype=np.int64)
        self.legal = legal
        self.priors = priors_full[legal].astype(np.float64)
        s = self.priors.sum()
        if s > 0:
            self.priors /= s
        else:                                   # net gave all legal moves 0; uniform
            self.priors[:] = 1.0 / len(legal)
        self.child_N = np.zeros(len(legal), dtype=np.float64)
        self.child_W = np.zeros(len(legal), dtype=np.float64)
        self.children = {}
        self.expanded = True
        return v

    def child_board(self, i):
        child = self.children.get(i)
        if child is None:
            nx = self.board.size_x
            x, y = index_to_move(int(self.legal[i]), nx)
            b = self.board.copy()
            b.move(x, y)
            child = Node(b)
            self.children[i] = child
        return child


class MCTS:
    def __init__(self, evaluator, cfg):
        self.ev = evaluator
        self.c_puct = cfg.c_puct
        self.sims = cfg.num_simulations
        self.alpha = cfg.dirichlet_alpha
        self.eps = cfg.dirichlet_eps
        self._rng = np.random.default_rng(cfg.seed)

    def _select(self, node):
        n_total = node.child_N.sum()
        sqrt_total = np.sqrt(n_total) if n_total > 0 else 1.0
        q = np.where(node.child_N > 0, node.child_W / np.maximum(node.child_N, 1), 0.0)
        u = self.c_puct * node.priors * sqrt_total / (1.0 + node.child_N)
        return int(np.argmax(q + u))

    def search(self, root_board, add_noise=True, time_budget=None, max_sims=None):
        """Run simulations from a copy of root_board. Returns (visit counts over
        all N*N actions, root). Stops after `max_sims` (default cfg sims) or when
        `time_budget` seconds elapse, whichever comes first."""
        root = Node(root_board.copy())
        root.expand(self.ev)
        if not root.is_terminal and add_noise and len(root.legal) > 1:
            noise = self._rng.dirichlet([self.alpha] * len(root.legal))
            root.priors = (1 - self.eps) * root.priors + self.eps * noise

        n_sims = max_sims if max_sims is not None else self.sims
        deadline = (time.time() + time_budget) if time_budget else None
        for sim in range(n_sims):
            if deadline and sim and time.time() > deadline:
                break
            node = root
            path = []                                    # list of (node, local_i)
            while node.expanded and not node.is_terminal:
                i = self._select(node)
                path.append((node, i))
                node = node.child_board(i)
            leaf_value = node.terminal_value if node.is_terminal else node.expand(self.ev)

            sign = -1.0                                  # parent-of-leaf sees -leaf_value
            for parent, i in reversed(path):
                parent.child_N[i] += 1
                parent.child_W[i] += sign * leaf_value
                sign = -sign

        nx = root_board.size_x
        counts = np.zeros(nx * root_board.size_y, dtype=np.float64)
        if root.legal is not None:
            counts[root.legal] = root.child_N
        return counts, root
