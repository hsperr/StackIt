"""How often does the exact same position repeat in the AlphaBeta shards, and
do the repeats agree on the labels?

Point of the exercise: supervised imitation stalls at ~0.48 top-1 and the value
head sits near 65% sign accuracy. Part of that wall may not be the net at all --
if the SAME position (same board, same side to move) is recorded once with
z=+1 and once with z=-1, no network can fit both. This script measures that
ceiling directly:

    value  ceiling = mean over positions of the majority z-sign share
    policy ceiling = mean over positions of the majority pi-move share

Position identity: planes are canonical to the player to move (see
alphazero/encoding.py), so the one-hot plane index per cell is a complete key --
25 bytes, exact, side-to-move included. No hashing collisions to worry about.

Weighting: "per position" below means per *sample*, i.e. a position seen 40
times counts 40 times, because that is what the training loss actually sees.
"""
import sys, glob
import numpy as np

CELLS = 25


def keys_and_labels(path):
    d = np.load(path)
    planes = d['planes']                                  # [n, 9, 5, 5]
    n = len(planes)
    codes = planes.reshape(n, 9, CELLS).argmax(1).astype(np.uint8)
    del planes
    return (np.ascontiguousarray(codes).view(f'|S{CELLS}').ravel(),
            d['z'].astype(np.float32),
            d['pi'].argmax(1).astype(np.int64))


def main():
    pats = sys.argv[1:] or ['research/ab_data/ab_shard*.npz']
    paths = sorted(p for pat in pats for p in glob.glob(pat))
    if not paths:
        sys.exit(f"no shards matched {pats}")

    K, Z, M = [], [], []
    for p in paths:
        k, z, m = keys_and_labels(p)
        K.append(k); Z.append(z); M.append(m)
        print(f"  {p}: {len(k):,} positions", flush=True)
    keys = np.concatenate(K); z = np.concatenate(Z); mv = np.concatenate(M)
    del K, Z, M

    uniq, inv = np.unique(keys, return_inverse=True)
    n, u = len(keys), len(uniq)
    cnt = np.bincount(inv, minlength=u)

    print(f"\n{len(paths)} shards, {n:,} positions, {u:,} unique "
          f"({u / n:.1%}), {n / u:.2f} copies each on average")

    # ---------------------------------------------------------- repeat sizes
    print("\nhow often a unique position is repeated:")
    edges = [1, 2, 3, 5, 10, 25, 100, 1 << 30]
    for lo, hi in zip(edges, edges[1:]):
        sel = (cnt >= lo) & (cnt < hi)
        if not sel.any():
            continue
        label = f"{lo}" if hi == lo + 1 else f"{lo}-{hi - 1}"
        print(f"  seen {label:>8}x : {sel.sum():>9,} unique "
              f"({cnt[sel].sum():>10,} samples, {cnt[sel].sum() / n:>5.1%})")
    print(f"  most repeated position appears {cnt.max():,} times")

    # ---------------------------------------------------------- value labels
    wins = np.bincount(inv, weights=(z > 0).astype(np.float64), minlength=u)
    loss = np.bincount(inv, weights=(z < 0).astype(np.float64), minlength=u)
    draw = cnt - wins - loss
    best_z = np.maximum.reduce([wins, loss, draw])
    conflict = (np.minimum(wins, loss) > 0)
    print(f"\nvalue (z) labels, from the mover's side:"
          f"\n  wins {wins.sum() / n:.1%}, losses {loss.sum() / n:.1%}, "
          f"draws {draw.sum() / n:.1%}"
          f"\n  unique positions with BOTH a win and a loss recorded: "
          f"{conflict.sum():,} ({conflict.sum() / u:.1%} of unique, "
          f"{cnt[conflict].sum() / n:.1%} of samples)"
          f"\n  best possible accuracy for any net (always pick the majority "
          f"label): {best_z.sum() / n:.1%}")

    # --------------------------------------------------------- policy labels
    pair = inv.astype(np.int64) * CELLS + mv
    pu, pc = np.unique(pair, return_counts=True)
    best_pi = np.zeros(u)
    np.maximum.at(best_pi, pu // CELLS, pc)
    nmoves = np.bincount(pu // CELLS, minlength=u)
    print(f"\npolicy (pi) labels, one-hot on AlphaBeta's move:"
          f"\n  unique positions where AlphaBeta was not always consistent: "
          f"{(nmoves > 1).sum():,} ({(nmoves > 1).sum() / u:.1%} of unique, "
          f"{cnt[nmoves > 1].sum() / n:.1%} of samples)"
          f"\n  best possible top-1 for any net: {best_pi.sum() / n:.1%}")

    # ----------------------------------------------- does repeating help fit?
    print("\nlabel agreement, split by how often the position repeats:")
    print(f"  {'repeats':>10} {'samples':>11} {'z ceiling':>10} {'pi ceiling':>11}")
    for lo, hi in zip(edges, edges[1:]):
        sel = (cnt >= lo) & (cnt < hi)
        if not sel.any():
            continue
        s = cnt[sel].sum()
        label = f"{lo}" if hi == lo + 1 else f"{lo}-{hi - 1}"
        print(f"  {label:>10} {s:>11,} {best_z[sel].sum() / s:>9.1%} "
              f"{best_pi[sel].sum() / s:>10.1%}")


if __name__ == "__main__":
    main()
