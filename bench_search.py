"""Search-performance benchmark for the root-level AlphaBeta engine.

Two things it measures, both from reproducible random mid-game positions:

  1. SEARCH DEPTH  — iterative-deepening alpha-beta on a fixed wall-clock
     budget: how deep it gets, and how many nodes/sec it pushes. This is the
     headline number to beat when we improve the engine.

  2. BOARD OPS     — make/undo + cascade cost in isolation. This is the work a
     bitboard / packed-int representation would attack, so it tells us the
     ceiling on what "free make/undo" could buy us.

    python3 bench_search.py                    # 5x5, 3s/position, 5 positions
    python3 bench_search.py --size 4 --time 2
    python3 bench_search.py --seed 7 --positions 8
"""
import argparse
import random
import time

from board import Board
from alphabeta import AlphaBeta


def random_position(size, plies, seed):
    """Play `plies` reproducible random legal moves from an empty board,
    stopping early if someone wins. Returns a realistic mid-game board."""
    rng = random.Random(seed)
    board = Board(size, size)
    for _ in range(plies):
        moves = board.possible_moves()
        if not moves or board.winning_player():
            break
        board.move(*rng.choice(moves))
    # hand back a fresh board with no undo history (search builds its own)
    return Board.from_custom_board(board.board, board.player, board.current_player)


def bench_board_ops(board, iters):
    """Average time for one move()+undo() pair, sampled across every legal
    move from `board` (so cheap no-cascade moves and big-cascade moves are
    both represented)."""
    moves = board.possible_moves()
    work = Board.from_custom_board(board.board, board.player, board.current_player)
    t0 = time.perf_counter()
    n = 0
    for _ in range(iters):
        for m in moves:
            work.move(*m)
            work.undo()
            n += 1
    dt = time.perf_counter() - t0
    return n, dt, n / dt


def bench_search(board, think, size):
    ab = AlphaBeta()
    t0 = time.perf_counter()
    move, score = ab.get_best_move(board, thinking_time=think, max_depth=1000)
    dt = time.perf_counter() - t0

    depth = int(ab.perft[-1][0]) if ab.perft else -1
    # perft stat strings look like "moves_made=1234 - quiet_moves=56 ..."
    nodes = 0
    for row in ab.perft:
        for tok in row[4].split(" - "):
            tok = tok.strip()
            if tok.startswith(("moves_made=", "quiet_moves=")):
                nodes += int(tok.split("=")[1])
    return {"move": move, "score": score, "depth": depth,
            "nodes": nodes, "time": dt, "nps": nodes / dt if dt else 0,
            "tt": len(ab.hashtable)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=5)
    ap.add_argument("--time", type=float, default=3.0, help="think seconds/position")
    ap.add_argument("--positions", type=int, default=5)
    ap.add_argument("--plies", type=int, default=None,
                    help="random moves to set up each position (default size*size//2)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    plies = args.plies if args.plies is not None else args.size * args.size // 2

    print(f"StackIt search benchmark — {args.size}x{args.size}, "
          f"{args.time}s/position, {args.positions} positions, {plies} setup plies\n")

    depths, npss = [], []
    print(f"{'pos':>3} {'depth':>6} {'nodes':>10} {'time':>6} {'nodes/s':>10} "
          f"{'tt':>7} {'best':>8}")
    for i in range(args.positions):
        board = random_position(args.size, plies, args.seed + i)
        r = bench_search(board, args.time, args.size)
        depths.append(r["depth"])
        npss.append(r["nps"])
        print(f"{i:>3} {r['depth']:>6} {r['nodes']:>10} {r['time']:>6.2f} "
              f"{r['nps']:>10,.0f} {r['tt']:>7} {str(r['move']):>8}")

    print(f"\nmean depth {sum(depths)/len(depths):.2f}   "
          f"min {min(depths)}  max {max(depths)}   "
          f"mean nodes/s {sum(npss)/len(npss):,.0f}")

    # Board-op microbenchmark on the first position.
    board = random_position(args.size, plies, args.seed)
    n, dt, ops = bench_board_ops(board, iters=2000)
    print(f"\nboard make+undo: {ops:,.0f} pairs/s  ({n:,} pairs in {dt:.2f}s)")
    print(f"  -> {1e9/ops:,.0f} ns per make+undo pair")


if __name__ == "__main__":
    main()
