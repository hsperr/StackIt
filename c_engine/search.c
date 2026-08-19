/* Faithful C port of the improved alphabeta.py (the version with the
 * quiescence stand-pat bound, ply-adjusted mate score, and QUIESCE_PLIES
 * constant). Same algorithm, same move ordering, same depths as the Python
 * — this is a like-for-like speed comparison, not a stronger engine. */
#include "search.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define QUIESCE_PLIES 6
#define MAX_PLY 1024
#define TT_BITS 22
#define TT_SIZE (1u << TT_BITS)
#define TT_MASK (TT_SIZE - 1u)

enum { TT_EMPTY = 0, TT_EXACT = 1, TT_UPPERBOUND = 2, TT_LOWERBOUND = 3 };

typedef struct {
    uint64_t key;
    int depth;
    int move;   /* cell index, -1 if none */
    int score;
    int flag;   /* TT_EMPTY / TT_EXACT / TT_UPPERBOUND / TT_LOWERBOUND */
} TTEntry;

/* Reading the clock at every node cost 42% of total runtime in a profile: at
 * ~4M nodes/s the search called mach_absolute_time about four million times a
 * second. Check it once per CLOCK_CHECK_MASK+1 calls instead - at that rate the
 * granularity is well under a millisecond, far finer than any time budget. */
#define CLOCK_CHECK_MASK 1023

typedef struct {
    double start_time;
    double allowed_time;
    int timed_out;
    unsigned clock_tick;
    int root_depth;
    TTEntry *tt;
    int killers[MAX_PLY][2];   /* -1 = no killer */
    long long history[MAX_CELLS];
    long long nodes;   /* full-width nodes: moves made at depth > 0 */
    long long qnodes;  /* quiescence nodes: moves made at depth <= 0 */
} SearchCtx;

static TTEntry *g_tt = NULL;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int time_over(SearchCtx *ctx) {
    if (ctx->timed_out) return 1;
    if (ctx->allowed_time <= 0) return 0;
    if (ctx->clock_tick++ & CLOCK_CHECK_MASK) return 0;
    if (now_seconds() - ctx->start_time > ctx->allowed_time) {
        ctx->timed_out = 1;
        return 1;
    }
    return 0;
}

/* Unthrottled check, for the few places that must be exact (the iterative
 * deepening loop, where one stale answer costs a whole extra iteration). */
static int time_over_exact(SearchCtx *ctx) {
    if (ctx->timed_out) return 1;
    if (ctx->allowed_time > 0 && now_seconds() - ctx->start_time > ctx->allowed_time) {
        ctx->timed_out = 1;
        return 1;
    }
    return 0;
}

/* Stable insertion sort, descending by rank. Insertion sort only shifts on a
 * STRICTLY lower rank, so equal-rank moves keep board_moves()'s original
 * (positionally pre-sorted) order - the search depends on that stability. */
static void stable_sort_moves(uint8_t *moves, long long *ranks, int n) {
    for (int i = 1; i < n; i++) {
        uint8_t km = moves[i];
        long long kr = ranks[i];
        int j = i - 1;
        while (j >= 0 && ranks[j] < kr) {
            moves[j + 1] = moves[j];
            ranks[j + 1] = ranks[j];
            j--;
        }
        moves[j + 1] = km;
        ranks[j + 1] = kr;
    }
}

/* Negamax with alpha-beta, PVS, TT, killers/history, and stand-pat
 * quiescence. Mirrors AlphaBeta._maxmimize() line for line, including its
 * quirk of storing `alpha` (not `best_score`) in the TT. */
static int negamax(Board *b, int alpha, int beta, int depth, SearchCtx *ctx, int *out_move) {
    if (time_over(ctx)) {
        *out_move = -1;
        return -1000000;
    }

    int original_alpha = alpha;

    int winner = board_winner(b);
    if (winner) {
        *out_move = -1;
        return -1000000 + (ctx->root_depth - depth);
    }

    int best_score = -10000000;
    int best_move = -1;

    int hash_move = -1;
    uint64_t key = b->zkey;
    TTEntry *slot = &ctx->tt[key & TT_MASK];
    if (slot->flag != TT_EMPTY && slot->key == key) {
        hash_move = slot->move;
        if (slot->depth >= depth) {
            if (slot->flag == TT_EXACT) {
                *out_move = hash_move;
                return slot->score;
            } else if (slot->flag == TT_LOWERBOUND) {
                if (slot->score > alpha) alpha = slot->score;
            } else if (slot->flag == TT_UPPERBOUND) {
                if (slot->score < beta) beta = slot->score;
            }
            if (alpha >= beta) {
                *out_move = hash_move;
                return slot->score;
            }
        }
    }

    uint8_t moves[MAX_CELLS];
    int nmoves;
    int in_quiescence = (depth <= 0);

    if (depth <= -QUIESCE_PLIES) {
        *out_move = -1;
        return board_eval(b);
    } else if (in_quiescence) {
        /* Stand-pat bound: standing pat (not exploding further) is always a
         * legal option, so its score lower-bounds this node. Without this the
         * side to move is forced to keep exploding even when it wouldn't. */
        int stand = board_eval(b);
        if (stand >= beta) {
            *out_move = -1;
            return stand;
        }
        if (stand > alpha) alpha = stand;
        best_score = stand;
        nmoves = board_attack_moves(b, moves);
    } else {
        nmoves = board_moves(b, moves);
    }

    if (nmoves == 0) {
        *out_move = -1;
        return board_eval(b);
    }

    /* Move ordering: hash move first, then killers for this ply, then
     * history; stable-sorted so equal-rank moves keep board_moves()'s order. */
    int ply = ctx->root_depth - depth;
    int k0 = -1, k1 = -1;
    if (ply >= 0 && ply < MAX_PLY) {
        k0 = ctx->killers[ply][0];
        k1 = ctx->killers[ply][1];
    }

    long long ranks[MAX_CELLS];
    for (int i = 0; i < nmoves; i++) {
        int m = moves[i];
        long long r;
        if (m == hash_move) {
            r = 1LL << 40;
        } else {
            r = ctx->history[m];
            if (m == k0) r += 1LL << 30;
            else if (m == k1) r += 1LL << 29;
        }
        ranks[i] = r;
    }
    stable_sort_moves(moves, ranks, nmoves);

    int first = 1;
    for (int i = 0; i < nmoves; i++) {
        int m = moves[i];
        Undo u;
        board_move(b, m, &u);
        if (depth <= 0) ctx->qnodes++;
        else ctx->nodes++;

        int child_move;
        int score;
        if (first) {
            score = -negamax(b, -beta, -alpha, depth - 1, ctx, &child_move);
        } else {
            /* Null-window scout; only pay for a full re-search if it beats
             * alpha and could still matter. */
            score = -negamax(b, -alpha - 1, -alpha, depth - 1, ctx, &child_move);
            if (!time_over(ctx) && score > alpha && score < beta) {
                score = -negamax(b, -beta, -alpha, depth - 1, ctx, &child_move);
            }
        }

        if (time_over(ctx)) {
            board_undo(b, &u);
            break;
        }
        board_undo(b, &u);
        first = 0;

        if (score > best_score) {
            best_score = score;
            best_move = m;
        }
        if (score > alpha) alpha = score;
        if (alpha >= beta) {
            if (ply >= 0 && ply < MAX_PLY) {
                if (ctx->killers[ply][0] != m) {
                    ctx->killers[ply][1] = ctx->killers[ply][0];
                    ctx->killers[ply][0] = m;
                }
            }
            ctx->history[m] += (long long)(depth + 7) * (depth + 7);
            break;
        }
    }

    if (!time_over(ctx)) {
        int flag;
        if (best_score <= original_alpha) flag = TT_UPPERBOUND;
        else if (best_score >= beta) flag = TT_LOWERBOUND;
        else flag = TT_EXACT;

        /* Deliberately mirrors the Python: the stored score is the final
         * `alpha`, not `best_score` (they can differ, e.g. a quiescence node
         * that stand-pats below original_alpha). Always-replace, no aging. */
        slot->key = key;
        slot->depth = depth;
        slot->move = best_move;
        slot->score = alpha;
        slot->flag = flag;
    }

    *out_move = best_move;
    return best_score;
}

SearchResult search_best_move(Board *b, double thinking_time, int max_depth) {
    SearchResult result;
    memset(&result, 0, sizeof(result));
    result.best_move = -1;
    result.depth = -1;

    double t0 = now_seconds();

    uint8_t root_moves[MAX_CELLS];
    int n = board_moves(b, root_moves);
    if (n == 0) {
        result.score = board_eval(b);
        result.secs = now_seconds() - t0;
        return result;
    }
    if (n == 1) {
        result.best_move = root_moves[0];
        result.score = 0;
        result.depth = 0;
        result.secs = now_seconds() - t0;
        return result;
    }

    if (!g_tt) {
        g_tt = (TTEntry *)malloc(sizeof(TTEntry) * TT_SIZE);
        if (!g_tt) {
            fprintf(stderr, "search: out of memory allocating transposition table\n");
            exit(1);
        }
    }
    memset(g_tt, 0, sizeof(TTEntry) * TT_SIZE); /* cleared per search() call, like the Python dict */

    SearchCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.start_time = t0;
    ctx.allowed_time = thinking_time;
    ctx.timed_out = 0;
    ctx.tt = g_tt;
    for (int i = 0; i < MAX_PLY; i++) {
        ctx.killers[i][0] = -1;
        ctx.killers[i][1] = -1;
    }

    int depth = 0;
    while (!time_over_exact(&ctx) && depth <= max_depth) {
        ctx.root_depth = depth;
        int move_out;
        int score = negamax(b, -1000000, 1000000, depth, &ctx, &move_out);

        depth++;

        if (!time_over_exact(&ctx)) {
            result.best_move = move_out;
            result.score = score;
            result.depth = depth - 1;
        }

        if (now_seconds() - ctx.start_time >= thinking_time) break;
    }

    result.nodes = ctx.nodes;
    result.qnodes = ctx.qnodes;
    result.secs = now_seconds() - t0;
    return result;
}
