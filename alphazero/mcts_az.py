"""PUCT / Gumbel Monte-Carlo Tree Search guided by the neural net.

A newly expanded leaf is evaluated *once* by the net: its policy seeds the child
priors and its value is backed up directly — no rollout. Interior selection uses
PUCT:

    U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))

Values are stored from each parent's own perspective, and the leaf value's sign
is flipped at every ply on the way back up (players alternate).

Two root policies:

* Classic PUCT (used when `time_budget` is set — wall-clock searches for the
  server engine and the strong-AlphaBeta benchmark). Optional Dirichlet noise.
* Gumbel root (default for fixed-sim searches — self-play and gate/eval). Uses
  Gumbel-top-k action sampling + Sequential Halving, and returns the "completed
  policy" (softmax of logits + sigma(completed Q)) as the improved target. This
  guarantees a policy improvement even at low sim counts, which plain PUCT+visit
  counts does not — the fix for the self-play gate plateau. See Danihelka et al.,
  "Policy Improvement by Planning with Gumbel", ICLR 2022.

Gumbel search has TWO distinct outputs, and conflating them is a bug: the
*completed policy* is the training target, while the *selected action* is the
Sequential-Halving survivor that should actually be PLAYED. Argmax of the
completed policy can differ from the survivor (an unvisited action's mixed value
can outrank a survivor's true Q), so playing it would execute a move Sequential
Halving already eliminated. `root.selected_action` holds the action to play.

`search()` always returns (policy_over_all_actions, root): a probability vector
of length N*N (0 on illegal moves, all-zero if the root is terminal). The root
also exposes `selected_action` (the action to play), plus child_N / child_W /
priors / children so engine.py and the dashboard can read visit counts and the
principal variation regardless of path.
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
                 "value", "legal", "priors", "child_N", "child_W", "children",
                 "selected_action")

    def __init__(self, board):
        self.board = board
        self.to_play = board.current_player
        self.expanded = False
        self.is_terminal = False
        self.terminal_value = 0.0
        self.value = 0.0            # net value estimate for this node (root perspective)
        self.legal = None
        self.priors = None
        self.child_N = None
        self.child_W = None
        self.children = None
        self.selected_action = None  # full action index the search chose to PLAY

    def expand(self, evaluator):
        tv = terminal_value(self.board)
        if tv is not None:
            self.is_terminal = True
            self.terminal_value = tv
            self.value = tv
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
        self.value = v
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
        self.gumbel_m = cfg.gumbel_m
        self.c_visit = cfg.gumbel_c_visit
        self.c_scale = cfg.gumbel_c_scale
        self.use_gumbel = getattr(cfg, "self_play_gumbel", True)
        self._rng = np.random.default_rng(cfg.seed)

    # -------------------------------------------------------------- selection
    def _select(self, node):
        n_total = node.child_N.sum()
        sqrt_total = np.sqrt(n_total) if n_total > 0 else 1.0
        q = np.where(node.child_N > 0, node.child_W / np.maximum(node.child_N, 1), 0.0)
        u = self.c_puct * node.priors * sqrt_total / (1.0 + node.child_N)
        return int(np.argmax(q + u))

    def _simulate_from(self, root, first_i):
        """One simulation forced through root child `first_i`, then PUCT below.
        Expands the reached leaf and backs its value up (including the root edge)."""
        path = [(root, first_i)]
        node = root.child_board(first_i)
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

    # ------------------------------------------------------------------- roots
    def _root(self, root_board, reuse_root):
        """Reuse a promoted subtree if given & usable, else build+expand fresh.
        The reused root must actually be THIS position (zkey encodes board + side
        to move) — a stale/mismatched root would silently search the wrong tree."""
        if (reuse_root is not None and reuse_root.expanded and not reuse_root.is_terminal
                and reuse_root.board.zkey == root_board.zkey):
            return reuse_root
        root = Node(root_board.copy())
        root.expand(self.ev)
        return root

    def _empty_policy(self, root_board):
        return np.zeros(root_board.size_x * root_board.size_y, dtype=np.float64)

    def _q(self, root):
        return np.where(root.child_N > 0,
                        root.child_W / np.maximum(root.child_N, 1), 0.0)

    def _q_completed(self, root):
        """Per-action Q with unvisited actions filled in by the mixed value
        v_mix (Gumbel paper eq. 'completed Q')."""
        N = root.child_N
        q = self._q(root)
        sum_N = N.sum()
        if sum_N > 0:
            prior = root.priors
            vis = N > 0
            denom = prior[vis].sum()
            wq = (prior[vis] * q[vis]).sum() / denom if denom > 0 else root.value
            v_mix = (root.value + sum_N * wq) / (1.0 + sum_N)
        else:
            v_mix = root.value
        return np.where(N > 0, q, v_mix)

    def _sigma(self, root, q_comp, max_visits=None):
        """mctx's `qtransform_completed_by_mix_value`: rescale the completed Q to
        [0, 1] FIRST, then scale by (c_visit + max_N) * c_scale.

        Both halves matter. Skipping the rescale (Q in [-1, 1]) and using
        c_scale=1.0 makes sigma ~20x larger than the prior logits, so the
        completed policy collapses to a near-one-hot pick driven by search noise
        (measured target entropy 0.005-0.76 out of 3.22 on 5x5) and Sequential
        Halving ignores its own Gumbel noise. mctx uses value_scale=0.1 on
        rescaled Q; c_scale in config.py now matches.

        `max_visits` is the largest visit count THIS search produced. It must not
        include visits carried over by tree reuse: mctx builds a fresh tree every
        move, so its max_N is bounded by the sim budget. Counting inherited visits
        inflates the multiplier — after a few reused plies max_N ran well past the
        budget, so sigma grew and the training target came out sharper than the
        algorithm intends. Verified against mctx `qtransforms.py`
        (value_scale=0.1, maxvisit_init=50.0, rescale_values=True)."""
        lo, hi = q_comp.min(), q_comp.max()
        span = hi - lo
        q = (q_comp - lo) / span if span > 1e-8 else np.zeros_like(q_comp)
        mv = root.child_N.max() if max_visits is None else max_visits
        return (self.c_visit + mv) * self.c_scale * q

    def _completed_policy(self, root, logits, max_visits=None):
        """Gumbel 'completed policy': softmax over legal of logits + sigma(q_comp),
        where visited actions use their search Q and unvisited use the mixed value
        v_mix. This is the improved-policy training target."""
        z = logits + self._sigma(root, self._q_completed(root), max_visits)
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        pi = np.zeros(root.board.size_x * root.board.size_y, dtype=np.float64)
        pi[root.legal] = p
        return pi

    def _gumbel_search(self, root, budget, add_noise, rng=None):
        """Gumbel-top-k + Sequential Halving at the root. Spends EXACTLY `budget`
        simulations (a truncated Sequential-Halving schedule, like mctx), stores
        the surviving action in `root.selected_action`, and returns the completed
        policy (the training target — a separate object from the played action).

        Exception: a single forced move is searched once, not `budget` times — the
        move is decided, so extra sims would only re-sample the same value."""
        k = len(root.legal)
        logits = np.log(np.clip(root.priors, 1e-9, None))
        # visits already on the root when tree reuse handed it to us; sigma's
        # max-visit term must count only the ones THIS search adds (see _sigma).
        base_N = root.child_N.copy()
        if k == 1:
            self._simulate_from(root, 0)
            root.selected_action = int(root.legal[0])
            return self._completed_policy(root, logits, (root.child_N - base_N).max())

        r = rng if rng is not None else self._rng
        g = r.gumbel(size=k) if add_noise else np.zeros(k)
        m = int(max(1, min(k, self.gumbel_m, budget)))
        cand = list(np.argsort(g + logits)[::-1][:m])
        n_phases = max(1, int(np.ceil(np.log2(m))))
        remaining = int(budget)                     # sims still to spend

        for phase in range(n_phases):
            n_cand = len(cand)
            phases_left = n_phases - phase
            # split the still-unspent budget evenly across the remaining phases,
            # then evenly across this phase's candidates; the last phase spends
            # whatever is left, so the schedule totals `budget` exactly.
            phase_budget = remaining if phases_left == 1 else max(n_cand, remaining // phases_left)
            phase_budget = min(phase_budget, remaining)
            per = max(1, phase_budget // n_cand)
            for a in cand:
                for _ in range(per):
                    if remaining <= 0:
                        break
                    self._simulate_from(root, a)
                    remaining -= 1
                if remaining <= 0:
                    break
            if n_cand <= 1:
                break
            scores = g + logits + self._sigma(root, self._q_completed(root),
                                              (root.child_N - base_N).max())
            cand = sorted(cand, key=lambda a: scores[a], reverse=True)[:max(1, n_cand // 2)]

        while remaining > 0:                         # spend any rounding remainder on the leader
            self._simulate_from(root, cand[0])
            remaining -= 1

        root.selected_action = int(root.legal[cand[0]])
        return self._completed_policy(root, logits, (root.child_N - base_N).max())

    def _puct_search(self, root, n_sims, deadline, add_noise, rng=None):
        """Classic PUCT loop (wall-clock / time-budget path). Returns normalized
        visit-count policy."""
        if add_noise and not root.is_terminal and len(root.legal) > 1:
            r = rng if rng is not None else self._rng
            noise = r.dirichlet([self.alpha] * len(root.legal))
            root.priors = (1 - self.eps) * root.priors + self.eps * noise
        for sim in range(n_sims):
            if deadline and sim and time.time() > deadline:
                break
            node = root
            path = []
            while node.expanded and not node.is_terminal:
                i = self._select(node)
                path.append((node, i))
                node = node.child_board(i)
            leaf_value = node.terminal_value if node.is_terminal else node.expand(self.ev)
            sign = -1.0
            for parent, i in reversed(path):
                parent.child_N[i] += 1
                parent.child_W[i] += sign * leaf_value
                sign = -sign
        counts = np.zeros(root.board.size_x * root.board.size_y, dtype=np.float64)
        counts[root.legal] = root.child_N
        s = counts.sum()
        return counts / s if s > 0 else counts

    def search(self, root_board, add_noise=True, time_budget=None, max_sims=None,
               reuse_root=None, rng=None):
        """Search from `root_board` (or a reused subtree). Returns (policy, root).

        * time_budget set -> classic PUCT for up to that many seconds (Dirichlet
          noise when add_noise); policy is normalized visit counts.
        * otherwise -> Gumbel root over `max_sims` (default cfg sims); Gumbel noise
          when add_noise; policy is the completed-policy improvement target.
        """
        root = self._root(root_board, reuse_root)
        if root.is_terminal or root.legal is None or len(root.legal) == 0:
            return self._empty_policy(root_board), root

        if time_budget is not None:
            n_sims = max_sims if max_sims is not None else self.sims
            deadline = time.time() + time_budget
            pi = self._puct_search(root, n_sims, deadline, add_noise, rng)
            # PUCT plays the most-visited root child.
            root.selected_action = int(root.legal[int(np.argmax(root.child_N))])
        elif self.use_gumbel:
            budget = max_sims if max_sims is not None else self.sims
            pi = self._gumbel_search(root, budget, add_noise, rng)  # sets root.selected_action
        else:
            # ABLATION: classic PUCT at fixed sim count -> visit-count policy target,
            # exactly the pre-Gumbel algorithm. Play the most-visited root child.
            n_sims = max_sims if max_sims is not None else self.sims
            pi = self._puct_search(root, n_sims, None, add_noise, rng)
            root.selected_action = int(root.legal[int(np.argmax(root.child_N))])
        return pi, root
