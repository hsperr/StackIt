#ifndef BOARD_H
#define BOARD_H
#include <stdint.h>

#define MAX_CELLS 64

typedef struct {
    int sx, sy, ncells;
    uint8_t val[MAX_CELLS];   /* chip count per cell */
    uint8_t own[MAX_CELLS];   /* 0 = nobody, 1 = player 1, 2 = player 2 */
    int current;              /* side to move: 1 or 2 */
    uint64_t zkey;            /* incremental Zobrist key */

    /* Incremental bookkeeping (kept in lockstep with val[]/own[] at every
     * mutation site so board_eval/board_winner can be O(1)). */
    int boxes[3];             /* boxes[o] = sum of val[i] over cells owned by o (0=unowned junk) */
    int owned[3];             /* owned[o] = count of cells owned by o */

    /* Per-cell geometry, precomputed once in board_init so board_moves'
     * sort never needs a div/mod. Indexed by cell index (y*sx+x). */
    uint8_t cellx[MAX_CELLS];
    uint8_t celly[MAX_CELLS];
    int emptyrank[MAX_CELLS];  /* rank used for an empty cell: 30 + |x-hx| + |y-hy| */
} Board;

typedef struct {
    uint8_t val[MAX_CELLS];
    uint8_t own[MAX_CELLS];
    int current;
    uint64_t zkey;
    int boxes[3];
    int owned[3];
} Undo;

void board_init(Board *b, int sx, int sy);
/* Set an ARBITRARY position (not reachable by replaying moves from the start).
 * val/own are ncells entries in ascending cell index; side is 1 or 2. All the
 * derived state (zkey, boxes, owned) is rebuilt from scratch. */
void board_set_position(Board *b, int sx, int sy, const uint8_t *val,
                        const uint8_t *own, int side);
int  board_moves(const Board *b, uint8_t *out);        /* legal moves, ordered; returns count */
int  board_attack_moves(const Board *b, uint8_t *out); /* quiescence moves, ordered; returns count */
void board_move(Board *b, int cell, Undo *u);          /* saves undo state into *u, then plays */
void board_undo(Board *b, const Undo *u);
int  board_winner(const Board *b);                     /* 0 = nobody, else 1 or 2 */
int  board_eval(const Board *b);                       /* boxes(side to move) - boxes(opponent) */

#endif
