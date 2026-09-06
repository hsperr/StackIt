# c_engine build. Owned files only: board.h/board.c/dump.c belong to this
# Makefile. search.c/main.c are owned by another agent -- do not add targets
# that reference them here; they can extend this file themselves.

CC = clang
CFLAGS = -std=c99 -O3 -Wall -Wextra

.PHONY: all libboard clean

all: dump

# Object file other translation units (e.g. search.c/main.c) link against.
libboard: board.o

board.o: board.c board.h
	$(CC) $(CFLAGS) -c board.c -o board.o

dump: dump.c board.o
	$(CC) $(CFLAGS) -o dump dump.c board.o

# Debug build: same board.c, but every board_move/board_undo call re-derives
# the Zobrist key from scratch and aborts on mismatch. Not part of the
# default targets; used only for the differential/zkey testing below.
board_verify.o: board.c board.h
	$(CC) $(CFLAGS) -DVERIFY_ZKEY -c board.c -o board_verify.o

dump-verify: dump.c board_verify.o
	$(CC) $(CFLAGS) -DVERIFY_ZKEY -o dump-verify dump.c board_verify.o

clean:
	rm -f board.o board_verify.o dump dump-verify
