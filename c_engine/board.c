/*
 * board.c -- C port of board.py (StackIt game rules).
 *
 * Faithful, line-for-line port of Board.move()/_throw_over()/_fields_to_throw()
 * from the Python source of record. The cascade is deliberately NOT rewritten
 * into a worklist: we replicate the "collect every cell >=5 across the whole
 * board, process that batch, rescan" ring structure exactly, because search
 * behaviour (and a later benchmark) depends on this being byte-identical to
 * the Python semantics, not just equivalent in the end state.
 */
#include "board.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* --- Zobrist hashing ----------------------------------------------------
 * piece[pos][owner][value] with piece[pos][0][0] == 0 so a truly empty cell
 * contributes nothing to the key. The value axis is sized to the full
 * uint8_t range (256) rather than the ~17 the Python side uses, because a
 * long cascade chain can transiently push a cell's value well past 16
 * before it topples again, and we never want that to walk off the table.
 * Any deterministic PRNG works here -- these keys do not need to match the
 * Python process's keys, only be internally consistent within this binary.
 */
#define ZVALS 256

typedef struct {
    uint64_t piece[MAX_CELLS][3][ZVALS];
    uint64_t side;
    int inited;
} ZTables;

static ZTables ZT;

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void init_zobrist(void) {
    if (ZT.inited) return;
    uint64_t state = 0x9E3779B97F4A7C15ULL; /* fixed seed -> reproducible keys */
    for (int pos = 0; pos < MAX_CELLS; pos++) {
        for (int owner = 0; owner < 3; owner++) {
            for (int v = 0; v < ZVALS; v++) {
                ZT.piece[pos][owner][v] = splitmix64(&state);
            }
        }
    }
    for (int pos = 0; pos < MAX_CELLS; pos++) {
        ZT.piece[pos][0][0] = 0;
    }
    ZT.side = splitmix64(&state);
    ZT.inited = 1;
}

/* Outside the VERIFY_ZKEY guard: board_set_position needs it in every build. */
static uint64_t zkey_from_scratch(const Board *b) {
    uint64_t z = 0;
    for (int i = 0; i < b->ncells; i++) {
        z ^= ZT.piece[i][b->own[i]][b->val[i]];
    }
    if (b->current == 2) z ^= ZT.side;
    return z;
}

#ifdef VERIFY_ZKEY
static void verify_zkey(const Board *b, const char *where) {
    uint64_t expect = zkey_from_scratch(b);
    if (expect != b->zkey) {
        fprintf(stderr, "ZKEY MISMATCH at %s: incremental=%llx recomputed=%llx\n",
                where, (unsigned long long)b->zkey, (unsigned long long)expect);
        abort();
    }
}
#endif

void board_init(Board *b, int sx, int sy) {
    init_zobrist();
    b->sx = sx;
    b->sy = sy;
    b->ncells = sx * sy;
    memset(b->val, 0, sizeof(b->val));
    memset(b->own, 0, sizeof(b->own));
    b->current = 1;
    /* every cell is (owner=0, val=0) -> piece[pos][0][0] == 0 for all pos;
       current==1 so the side key is not folded in (matches Python's
       _compute_zkey, which only XORs side when current_player == 2). */
    b->zkey = 0;

    /* All cells start unowned with val 0: boxes/owned are trivial. */
    b->boxes[0] = 0; b->boxes[1] = 0; b->boxes[2] = 0;
    b->owned[0] = b->ncells; b->owned[1] = 0; b->owned[2] = 0;

    int hx = sx / 2, hy = sy / 2;
    for (int y = 0; y < sy; y++) {
        for (int x = 0; x < sx; x++) {
            int idx = y * sx + x;
            b->cellx[idx] = (uint8_t)x;
            b->celly[idx] = (uint8_t)y;
            int dx = x > hx ? x - hx : hx - x;
            int dy = y > hy ? y - hy : hy - y;
            b->emptyrank[idx] = 30 + dx + dy;
        }
    }
}

void board_set_position(Board *b, int sx, int sy, const uint8_t *val,
                        const uint8_t *own, int side) {
    board_init(b, sx, sy);                  /* geometry tables + zobrist tables */
    b->current = side;
    for (int i = 0; i < b->ncells; i++) {
        b->val[i] = val[i];
        b->own[i] = own[i];
    }
    /* Rebuild the incremental bookkeeping the mutation sites normally maintain. */
    b->boxes[0] = b->boxes[1] = b->boxes[2] = 0;
    b->owned[0] = b->owned[1] = b->owned[2] = 0;
    for (int i = 0; i < b->ncells; i++) {
        b->boxes[b->own[i]] += b->val[i];
        b->owned[b->own[i]] += 1;
    }
    b->zkey = zkey_from_scratch(b);
}

int board_moves(const Board *b, uint8_t *out) {
    /* hx/hy no longer needed here -- emptyrank[] is precomputed in board_init. */
    int cur = b->current;
    int sx = b->sx, sy = b->sy;

    int rank10[MAX_CELLS];
    int cell[MAX_CELLS];
    int n = 0;

    for (int y = 0; y < sy; y++) {
        for (int x = 0; x < sx; x++) {
            int idx = y * sx + x;
            int p = b->own[idx];
            if (!p || p == cur) {
                int field = b->val[idx];
                int r = field ? field * 10 : b->emptyrank[idx]; /* scaled: (3 + dist/10) * 10 */
                rank10[n] = r;
                cell[n] = idx;
                n++;
            }
        }
    }

    /* insertion sort on (rank10, x, y) ascending -- n is small (<= MAX_CELLS)
       so O(n^2) is fine and keeps this allocation-free. cellx/celly are
       precomputed per-cell tables, so the comparison is pure array lookups
       (no div/mod inside the sort). */
    for (int i = 1; i < n; i++) {
        int rr = rank10[i], cc = cell[i];
        int cx = b->cellx[cc], cy = b->celly[cc];
        int j = i - 1;
        while (j >= 0) {
            int jx = b->cellx[cell[j]], jy = b->celly[cell[j]];
            int greater = (rank10[j] > rr) ||
                          (rank10[j] == rr && (jx > cx || (jx == cx && jy > cy)));
            if (!greater) break;
            rank10[j + 1] = rank10[j];
            cell[j + 1] = cell[j];
            j--;
        }
        rank10[j + 1] = rr;
        cell[j + 1] = cc;
    }

    for (int i = 0; i < n; i++) out[i] = (uint8_t)cell[i];
    return n;
}

int board_attack_moves(const Board *b, uint8_t *out) {
    int cur = b->current;
    int n = 0;
    for (int idx = 0; idx < b->ncells; idx++) {
        if (b->own[idx] == cur && b->val[idx] == 4) {
            out[n++] = (uint8_t)idx;
        }
    }
    return n;
}

/* Owner of `pos` is unchanged by exploding -- only its value drops by 4, so
   the boxes[] bucket for its (unchanged) owner just needs the delta. */
static inline void throw_explode(Board *b, int pos) {
    int po = b->own[pos];
    b->zkey ^= ZT.piece[pos][po][b->val[pos]];
    b->val[pos] -= 4;
    b->zkey ^= ZT.piece[pos][po][b->val[pos]];
    b->boxes[po] -= 4;
}

/* Neighbour gains one chip and is claimed by `cur`, whatever it was before:
   subtract its old (owner,val) contribution, mutate, add the new one back.
   This is correct whether or not the neighbour already belonged to cur. */
static inline void throw_claim(Board *b, int np, int cur) {
    int old_o = b->own[np];
    int old_v = b->val[np];
    b->boxes[old_o] -= old_v;
    b->owned[old_o]--;

    b->zkey ^= ZT.piece[np][old_o][old_v];
    b->val[np] = (uint8_t)(old_v + 1);
    b->own[np] = (uint8_t)cur;
    b->zkey ^= ZT.piece[np][cur][b->val[np]];

    b->boxes[cur] += b->val[np];
    b->owned[cur]++;
}

static void throw_over(Board *b, int x, int y, int sx, int sy, int cur) {
    int pos = y * sx + x;

    throw_explode(b, pos);

    if (y + 1 < sy) throw_claim(b, pos + sx, cur);
    if (y - 1 >= 0) throw_claim(b, pos - sx, cur);
    if (x + 1 < sx) throw_claim(b, pos + 1, cur);
    if (x - 1 >= 0) throw_claim(b, pos - 1, cur);
}

static int fields_to_throw(const Board *b, int *out) {
    int n = 0;
    for (int idx = 0; idx < b->ncells; idx++) {
        if (b->val[idx] >= 5) out[n++] = idx;
    }
    return n;
}

void board_move(Board *b, int cell, Undo *u) {
    memcpy(u->val, b->val, sizeof(b->val));
    memcpy(u->own, b->own, sizeof(b->own));
    u->current = b->current;
    u->zkey = b->zkey;
    u->boxes[0] = b->boxes[0]; u->boxes[1] = b->boxes[1]; u->boxes[2] = b->boxes[2];
    u->owned[0] = b->owned[0]; u->owned[1] = b->owned[1]; u->owned[2] = b->owned[2];

    int sx = b->sx, sy = b->sy;
    int x = cell % sx, y = cell / sx;
    int cur = b->current;

    int old_own = b->own[cell];
    int old_val = b->val[cell];
    b->boxes[old_own] -= old_val;
    b->owned[old_own]--;

    b->zkey ^= ZT.piece[cell][old_own][old_val];
    if (old_val == 0) {
        b->val[cell] = (uint8_t)(old_val + 3); /* Board.INITAL_BOX_INCREASE */
    } else {
        b->val[cell] = (uint8_t)(old_val + 1);
    }
    b->own[cell] = (uint8_t)cur;
    b->zkey ^= ZT.piece[cell][cur][b->val[cell]];

    b->boxes[cur] += b->val[cell];
    b->owned[cur]++;

    if (b->val[cell] >= 5) {
        throw_over(b, x, y, sx, sy, cur);

        int fields[MAX_CELLS];
        int nf = fields_to_throw(b, fields);
        while (nf > 0) {
            for (int i = 0; i < nf; i++) {
                int fidx = fields[i];
                throw_over(b, fidx % sx, fidx / sx, sx, sy, cur);
            }
            nf = fields_to_throw(b, fields);
        }
    }

    b->zkey ^= ZT.side;
    b->current = (cur == 1) ? 2 : 1;

#ifdef VERIFY_ZKEY
    verify_zkey(b, "board_move");
#endif
}

void board_undo(Board *b, const Undo *u) {
    memcpy(b->val, u->val, sizeof(b->val));
    memcpy(b->own, u->own, sizeof(b->own));
    b->current = u->current;
    b->zkey = u->zkey;
    b->boxes[0] = u->boxes[0]; b->boxes[1] = u->boxes[1]; b->boxes[2] = u->boxes[2];
    b->owned[0] = u->owned[0]; b->owned[1] = u->owned[1]; b->owned[2] = u->owned[2];

#ifdef VERIFY_ZKEY
    verify_zkey(b, "board_undo");
#endif
}

int board_winner(const Board *b) {
    if (b->owned[1] == b->ncells) return 1;
    if (b->owned[2] == b->ncells) return 2;
    return 0;
}

int board_eval(const Board *b) {
    int cur = b->current;
    int other = 3 - cur;
    return b->boxes[cur] - b->boxes[other];
}
