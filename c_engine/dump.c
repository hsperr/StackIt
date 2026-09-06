/*
 * dump.c -- differential-testing CLI for board.c.
 *
 * Usage: ./dump <sx> <sy> "<moves>"
 *   <moves> is a space-separated list of "x,y" pairs applied in order from
 *   the initial board. May be empty (omit the third argument, or pass "").
 *
 * Prints exactly seven lines describing the resulting position; see the
 * project's header contract / difftest.py for the exact format.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "board.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <sx> <sy> \"<moves>\"\n", argv[0]);
        return 1;
    }

    int sx = atoi(argv[1]);
    int sy = atoi(argv[2]);
    const char *movestr = (argc >= 4) ? argv[3] : "";

    if (sx <= 0 || sy <= 0 || sx * sy > MAX_CELLS) {
        fprintf(stderr, "invalid board size %dx%d (max %d cells)\n", sx, sy, MAX_CELLS);
        return 1;
    }

    Board b;
    board_init(&b, sx, sy);

    char buf[8192];
    size_t len = strlen(movestr);
    if (len >= sizeof(buf)) len = sizeof(buf) - 1;
    memcpy(buf, movestr, len);
    buf[len] = '\0';

    char *saveptr = NULL;
    char *tok = strtok_r(buf, " ", &saveptr);
    while (tok) {
        int x, y;
        if (sscanf(tok, "%d,%d", &x, &y) == 2) {
            if (x < 0 || x >= sx || y < 0 || y >= sy) {
                fprintf(stderr, "move out of bounds: %s\n", tok);
                return 1;
            }
            int cell = y * sx + x;
            Undo u;
            board_move(&b, cell, &u);
        }
        tok = strtok_r(NULL, " ", &saveptr);
    }

    printf("val");
    for (int i = 0; i < b.ncells; i++) printf(" %d", b.val[i]);
    printf("\n");

    printf("own");
    for (int i = 0; i < b.ncells; i++) printf(" %d", b.own[i]);
    printf("\n");

    printf("current %d\n", b.current);
    printf("winner %d\n", board_winner(&b));
    printf("eval %d\n", board_eval(&b));

    uint8_t moves[MAX_CELLS];
    int nm = board_moves(&b, moves);
    printf("moves");
    for (int i = 0; i < nm; i++) printf(" %d,%d", moves[i] % sx, moves[i] / sx);
    printf("\n");

    uint8_t attacks[MAX_CELLS];
    int na = board_attack_moves(&b, attacks);
    printf("attack");
    for (int i = 0; i < na; i++) printf(" %d,%d", attacks[i] % sx, attacks[i] / sx);
    printf("\n");

    return 0;
}
