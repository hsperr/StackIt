"""Elo ratings for model versions on one absolute scale.

The gate win rate only says "better than the immediately-previous model" — it
can't tell you whether version 12 beats version 3. To get an absolute picture we
record every match between participants (model versions + `random` + `alphabeta`)
in a ResultBook and fit a single set of Elo ratings to *all* of them at once.

We use the Bradley-Terry model solved by the standard MM (minorization-
maximization) iteration — the same idea behind BayesElo. Ratings are anchored so
`random` = 0, giving the scale a fixed meaning: refining old versions' ratings
as more games accumulate, without the drift that online per-game Elo updates
suffer from. Draws count as half a win to each side.
"""
import math

SCALE = 400.0   # Elo points per factor-of-10 in Bradley-Terry strength


class ResultBook:
    """Accumulates pairwise match results between named participants."""

    def __init__(self):
        self.d = {}   # (a, b) with a,b sorted by str -> [wins_a, wins_b, draws]

    @staticmethod
    def _key(p, q):
        return tuple(sorted((p, q), key=str))

    def add_match(self, p, q, wins_p, wins_q, draws):
        if wins_p + wins_q + draws == 0:
            return          # a 0-game "match" would still collect prior_draws in the fit
        a, b = self._key(p, q)
        rec = self.d.setdefault((a, b), [0, 0, 0])
        if a == p:
            rec[0] += wins_p
            rec[1] += wins_q
        else:
            rec[0] += wins_q
            rec[1] += wins_p
        rec[2] += draws

    def participants(self):
        s = set()
        for a, b in self.d:
            s.add(a)
            s.add(b)
        return s

    def games_played(self):
        """Total games each participant has played (Elo trustworthiness)."""
        g = {}
        for (a, b), (wa, wb, d) in self.d.items():
            n = wa + wb + d
            g[a] = g.get(a, 0) + n
            g[b] = g.get(b, 0) + n
        return g

    def to_list(self):
        return [{"a": a, "b": b, "wa": r[0], "wb": r[1], "d": r[2]}
                for (a, b), r in self.d.items()]

    @classmethod
    def from_list(cls, lst):
        bk = cls()
        for e in lst:
            bk.d[(e["a"], e["b"])] = [e["wa"], e["wb"], e["d"]]
        return bk


def compute_elo(book, anchor="random", anchor_elo=0.0, iters=250, prior_draws=0.0):
    """Fit Bradley-Terry strengths to all recorded matches, return {id: elo}.

    `prior_draws` adds that many virtual draws to every played pair — a mild
    Bayesian shrink that stops a 16-0 sweep from demanding an infinite (and
    wildly noisy) rating gap, so Elo can't run away."""
    parts = list(book.participants())
    if not parts:
        return {}

    wins = {p: 0.0 for p in parts}      # wins (draw = 0.5) per participant
    npair = {}                          # games played between each ordered pair
    for (a, b), (wa, wb, d) in book.d.items():
        d = d + prior_draws
        wins[a] += wa + 0.5 * d
        wins[b] += wb + 0.5 * d
        n = wa + wb + d
        npair[(a, b)] = npair.get((a, b), 0) + n
        npair[(b, a)] = npair.get((b, a), 0) + n

    gamma = {p: 1.0 for p in parts}
    for _ in range(iters):
        ng = {}
        for i in parts:
            denom = 0.0
            for j in parts:
                if i == j:
                    continue
                n = npair.get((i, j), 0)
                if n:
                    denom += n / (gamma[i] + gamma[j])
            ng[i] = wins[i] / denom if (wins[i] > 0 and denom > 0) else gamma[i]
        # renormalize to geometric mean 1 so the scale can't drift
        logs = [math.log(v) for v in ng.values() if v > 0]
        m = math.exp(sum(logs) / len(logs)) if logs else 1.0
        gamma = {p: (v / m if v > 0 else 1e-9) for p, v in ng.items()}

    elo = {p: SCALE * math.log10(max(gamma[p], 1e-9)) for p in parts}
    if anchor in elo:
        shift = anchor_elo - elo[anchor]
        elo = {p: e + shift for p, e in elo.items()}
    return elo
