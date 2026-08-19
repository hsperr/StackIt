/* Two modes.
 *
 * One-shot:  ./stackit <sx> <sy> "<moves>" <seconds> [max_depth]
 *   <moves> is a space-separated list of "x,y" pairs applied from the initial
 *   board (may be empty). Prints one JSON line for the resulting position.
 *
 * Serve:     ./stackit --serve
 *   Reads request lines from stdin and writes one JSON line per request. Used
 *   by alphazero/arena_eval.py so a match pays one process spawn instead of one
 *   per move. Each request is an ARBITRARY position, not a move list:
 *     <sx> <sy> <side> <seconds> <max_depth> <v:o> <v:o> ...
 *   with one "value:owner" pair per cell in ascending index order (idx = y*sx+x),
 *   owner 0/1/2 and side to move 1/2. A malformed line answers with
 *   {"move":null,"error":"..."} and the loop continues — a bad request must not
 *   kill an engine that a running match is talking to. */
#include "board.h"
#include "search.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_result(const SearchResult *res, int sx) {
    if (res->best_move < 0) {
        printf("{\"move\":null,\"score\":%d,\"depth\":%d,\"nodes\":%lld,"
               "\"qnodes\":%lld,\"secs\":%.4f}\n",
               res->score, res->depth, res->nodes, res->qnodes, res->secs);
    } else {
        printf("{\"move\":[%d,%d],\"score\":%d,\"depth\":%d,\"nodes\":%lld,"
               "\"qnodes\":%lld,\"secs\":%.4f}\n",
               res->best_move % sx, res->best_move / sx,
               res->score, res->depth, res->nodes, res->qnodes, res->secs);
    }
    fflush(stdout);
}

static void serve_error(const char *msg) {
    printf("{\"move\":null,\"error\":\"%s\"}\n", msg);
    fflush(stdout);
}

/* One request line -> one JSON line. Returns 0 always; errors are reported to
 * the caller rather than exiting, so the stream survives a bad request. */
static void serve_line(char *line) {
    char *sp = NULL;
    char *t_sx = strtok_r(line, " \t\n", &sp);
    char *t_sy = strtok_r(NULL, " \t\n", &sp);
    char *t_side = strtok_r(NULL, " \t\n", &sp);
    char *t_secs = strtok_r(NULL, " \t\n", &sp);
    char *t_depth = strtok_r(NULL, " \t\n", &sp);
    if (!t_sx || !t_sy || !t_side || !t_secs || !t_depth) {
        serve_error("short request");
        return;
    }
    int sx = atoi(t_sx), sy = atoi(t_sy), side = atoi(t_side);
    double secs = atof(t_secs);
    int max_depth = atoi(t_depth);
    if (sx <= 0 || sy <= 0 || sx * sy > MAX_CELLS || (side != 1 && side != 2)) {
        serve_error("bad header");
        return;
    }

    int ncells = sx * sy;
    uint8_t val[MAX_CELLS], own[MAX_CELLS];
    for (int i = 0; i < ncells; i++) {
        char *tok = strtok_r(NULL, " \t\n", &sp);
        int v, o;
        if (!tok || sscanf(tok, "%d:%d", &v, &o) != 2 || v < 0 || v > 255 || o < 0 || o > 2) {
            serve_error("bad cell");
            return;
        }
        val[i] = (uint8_t)v;
        own[i] = (uint8_t)o;
    }

    Board b;
    board_set_position(&b, sx, sy, val, own, side);
    SearchResult res = search_best_move(&b, secs, max_depth);
    print_result(&res, sx);
}

static int serve_loop(void) {
    /* Requests are one line each; MAX_CELLS cells at "vvv:o " plus the header
     * fits comfortably, but size generously and reject overlong lines. */
    static char line[8192];
    while (fgets(line, sizeof(line), stdin)) {
        if (!strchr(line, '\n') && !feof(stdin)) {
            int c;
            while ((c = fgetc(stdin)) != EOF && c != '\n') { }
            serve_error("request too long");
            continue;
        }
        serve_line(line);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--serve") == 0) {
        return serve_loop();
    }
    if (argc < 5) {
        fprintf(stderr, "usage: %s <sx> <sy> \"<moves>\" <seconds> [max_depth]\n"
                        "       %s --serve\n", argv[0], argv[0]);
        return 1;
    }

    int sx = atoi(argv[1]);
    int sy = atoi(argv[2]);
    const char *moves_str = argv[3];
    double seconds = atof(argv[4]);
    int max_depth = (argc >= 6) ? atoi(argv[5]) : 1000000;

    if (sx <= 0 || sy <= 0 || sx * sy > MAX_CELLS) {
        fprintf(stderr, "invalid board size %dx%d (max %d cells)\n", sx, sy, MAX_CELLS);
        return 1;
    }

    Board b;
    board_init(&b, sx, sy);

    /* Replay the move list, "x,y" pairs separated by whitespace. */
    char *buf = strdup(moves_str);
    if (!buf) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    char *saveptr = NULL;
    char *tok = strtok_r(buf, " \t\n", &saveptr);
    while (tok) {
        int x, y;
        if (sscanf(tok, "%d,%d", &x, &y) != 2) {
            fprintf(stderr, "bad move token '%s'\n", tok);
            free(buf);
            return 1;
        }
        if (x < 0 || x >= sx || y < 0 || y >= sy) {
            fprintf(stderr, "move '%s' out of bounds for %dx%d board\n", tok, sx, sy);
            free(buf);
            return 1;
        }
        int cell = y * sx + x;
        Undo u;
        board_move(&b, cell, &u);
        tok = strtok_r(NULL, " \t\n", &saveptr);
    }
    free(buf);

    SearchResult res = search_best_move(&b, seconds, max_depth);
    print_result(&res, sx);
    return 0;
}
