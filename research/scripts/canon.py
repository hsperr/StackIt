"""Canonical orientation instead of 8-fold augmentation.

Idea: map every board to a single canonical member of its dihedral orbit, train
only on those, and at inference rotate in, ask the net, rotate the move back.
Rotation invariance then holds BY CONSTRUCTION rather than being something the
net has to learn -- which is exactly what it has been failing to learn (52%
self-disagreement under rotation despite 8-fold augmentation).

The catch this script measures: AlphaBeta breaks ties by move index, which is not
rotation-equivariant. So two rotations of one position can carry labels that are
NOT rotations of each other. Canonicalising merges them, and those merges become
contradictory targets that no network can fit. If that fraction is large, the
idea costs more than it pays.

Canonical form = the dihedral image whose encoded planes are lexicographically
smallest. Ties mean the images are byte-identical, so the choice is irrelevant.
"""
import sys
sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')
import numpy as np
from collections import defaultdict
from alphazero.encoding import _dihedral_ops, _apply_planes, _apply_grid

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else None
N = 5

d = np.load(src)
P, PI = d["planes"].astype(np.float32), d["pi"].astype(np.float32)
Z, OWN = d["z"].astype(np.float32), d["own"].astype(np.int64)
OPS = _dihedral_ops(True)
print(f"{len(P)} positions, {len(OPS)} dihedral ops", flush=True)

cP = np.empty_like(P)
cPI = np.empty_like(PI)
cOWN = np.empty_like(OWN)
for i in range(len(P)):
    best_key = None
    for k, flip in OPS:
        t = _apply_planes(P[i], k, flip)
        key = t.tobytes()
        if best_key is None or key < best_key:
            best_key, best_t, best_op = key, t, (k, flip)
    k, flip = best_op
    cP[i] = best_t
    cPI[i] = _apply_grid(PI[i].reshape(N, N), k, flip).reshape(-1)
    cOWN[i] = _apply_grid(OWN[i].reshape(N, N), k, flip).reshape(-1)
    if (i + 1) % 50000 == 0:
        print(f"  canonicalised {i+1}/{len(P)}", flush=True)

# How much did the orbit collapse, and at what cost in contradictory labels?
groups = defaultdict(list)
for i in range(len(cP)):
    groups[cP[i].tobytes()].append(i)
dup = {k: v for k, v in groups.items() if len(v) > 1}
conflict_groups = conflict_rows = 0
for k, idx in dup.items():
    labels = {int(cPI[i].argmax()) for i in idx}
    if len(labels) > 1:
        conflict_groups += 1
        conflict_rows += len(idx)

raw_distinct = len({P[i].tobytes() for i in range(len(P))})
print(f"\ndistinct BEFORE canonicalisation : {raw_distinct}")
print(f"distinct AFTER  canonicalisation : {len(groups)}")
print(f"  collapse ratio                 : {raw_distinct/max(len(groups),1):.2f}x")
print(f"groups with >1 member            : {len(dup)}")
print(f"  of those, contradictory labels : {conflict_groups} "
      f"({conflict_groups/max(len(dup),1):.1%} of duplicate groups)")
print(f"  rows sitting in a conflict     : {conflict_rows} "
      f"({conflict_rows/len(P):.2%} of the dataset)")

if out:
    np.savez_compressed(out, planes=cP, pi=cPI, z=Z, own=cOWN)
    print(f"\nwrote {out}", flush=True)
