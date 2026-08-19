#ifndef SEARCH_H
#define SEARCH_H

#include "board.h"

typedef struct {
    int best_move;      /* cell index, -1 if none */
    int score;
    int depth;           /* last COMPLETED depth */
    long long nodes;     /* full-width nodes (moves made at depth > 0) */
    long long qnodes;    /* quiescence nodes (moves made at depth <= 0) */
    double secs;
} SearchResult;

SearchResult search_best_move(Board *b, double thinking_time, int max_depth);

#endif
