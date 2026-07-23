"""Board <-> neural-net tensor encoding, action indexing, and symmetry augmentation.

Canonical form: the board is always encoded from the perspective of the player
to move ("mine" vs "theirs"), so the net only ever has to reason about "my
turn". This is the single most bug-prone part of AlphaZero — get the perspective
or the value sign wrong and learning silently fails — so it lives here alone,
covered by tests.

Planes (all binary), for an NxN board -> shape [9, N, N]:
    0..3  : my cell with block-count 1,2,3,4
    4..7  : opponent cell with block-count 1,2,3,4
    8     : empty cell
After any move fully resolves, every cell holds 0..4 blocks (5+ always topples),
so these one-hot count planes are exhaustive.
"""
import numpy as np

NUM_PLANES = 9


def encode(board):
    """Board -> float32 array [9, N, N], canonical to board.current_player."""
    ny, nx = board.size_y, board.size_x
    planes = np.zeros((NUM_PLANES, ny, nx), dtype=np.float32)
    cur = board.current_player
    grid, owners = board.board, board.player
    for y in range(ny):
        brow, prow = grid[y], owners[y]
        for x in range(nx):
            o = prow[x]
            if o == 0:
                planes[8, y, x] = 1.0
                continue
            c = brow[x]
            c = 1 if c < 1 else (4 if c > 4 else c)   # clamp into 1..4
            base = 0 if o == cur else 4
            planes[base + c - 1, y, x] = 1.0
    return planes


def action_index(x, y, size_x):
    return y * size_x + x


def index_to_move(idx, size_x):
    return idx % size_x, idx // size_x       # (x, y)


def num_actions(size_x, size_y):
    return size_x * size_y


def legal_mask(board):
    """Boolean array of length N*N, True where a move is legal."""
    nx = board.size_x
    mask = np.zeros(board.size_x * board.size_y, dtype=bool)
    for (x, y) in board.possible_moves():
        mask[action_index(x, y, nx)] = True
    return mask


# --------------------------------------------------------------- symmetries
def _dihedral_ops(square):
    """Yield (name, transform) over the symmetry group. Full D4 (8 ops) for
    square boards; just identity + horizontal flip for non-square."""
    if square:
        return [(k, flip) for k in range(4) for flip in (False, True)]
    return [(0, False), (0, True)]


def _apply_planes(planes, k, flip):
    out = np.rot90(planes, k=k, axes=(1, 2))
    if flip:
        out = out[:, :, ::-1]
    return np.ascontiguousarray(out)


def _apply_grid(grid, k, flip):
    out = np.rot90(grid, k=k)
    if flip:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


def augment(planes, pi, size_x, size_y):
    """Given encoded `planes` [9,N,N] and policy target `pi` (flat len N*N),
    yield (planes', pi') for every board symmetry. The SAME dihedral op is
    applied to both so the (state, policy) pairing stays consistent."""
    square = size_x == size_y
    pi_grid = np.asarray(pi, dtype=np.float32).reshape(size_y, size_x)
    seen = set()
    for k, flip in _dihedral_ops(square):
        p = _apply_planes(planes, k, flip)
        g = _apply_grid(pi_grid, k, flip)
        key = p.tobytes()
        if key in seen:            # skip duplicate orientations (symmetric states)
            continue
        seen.add(key)
        yield p, g.reshape(-1)
