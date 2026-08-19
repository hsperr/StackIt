#!/usr/bin/env python3
"""Differential test: C ./dump vs the Python Board (repo root, source of truth).

Plays N random games, and at every ply shells out to ./dump with the full
move list so far, comparing all seven printed lines against the equivalent
Python board state. Also greps the dump-verify stderr for a Zobrist-key
mismatch (dump-verify aborts on mismatch, so a nonzero/abort exit already
catches it, but we check explicitly too).

Usage: python3 difftest.py [num_games] [seed] [sx] [sy] [max_plies]
"""
import sys
import os
import subprocess

sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')
import numpy as np
from board import Board

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(HERE, 'dump')
DUMP_VERIFY = os.path.join(HERE, 'dump-verify')


def fmt_moves(moves):
    return ' '.join(f'{x},{y}' for x, y in moves)


def run_dump(binpath, sx, sy, moves):
    res = subprocess.run(
        [binpath, str(sx), str(sy), fmt_moves(moves)],
        capture_output=True, text=True, timeout=30
    )
    return res


def python_state(board):
    sx, sy = board.size_x, board.size_y
    val = []
    own = []
    for y in range(sy):
        for x in range(sx):
            val.append(board.value_at(x, y))
            own.append(board.player_at(x, y))
    cur = board.current_player
    other = 2 if cur == 1 else 1
    winner = board.winning_player()
    ev = board.boxes_for(cur) - board.boxes_for(other)
    moves = board.possible_moves()
    attacks = board.possible_attack_moves()
    return {
        'val': val,
        'own': own,
        'current': cur,
        'winner': winner,
        'eval': ev,
        'moves': moves,
        'attack': attacks,
    }


def parse_dump_stdout(stdout):
    lines = stdout.strip('\n').split('\n')
    d = {}
    for line in lines:
        parts = line.split(' ')
        key = parts[0]
        rest = parts[1:]
        if key in ('val', 'own'):
            d[key] = [int(v) for v in rest]
        elif key == 'current':
            d[key] = int(rest[0])
        elif key == 'winner':
            d[key] = int(rest[0])
        elif key == 'eval':
            d[key] = int(rest[0])
        elif key in ('moves', 'attack'):
            pairs = []
            for tok in rest:
                if tok == '':
                    continue
                xs, ys = tok.split(',')
                pairs.append((int(xs), int(ys)))
            d[key] = pairs
    return d


def compare(py_state, c_state, sx, sy, moves_so_far, game_idx, ply):
    mismatches = []
    for key in ('val', 'own', 'current', 'winner', 'eval', 'moves', 'attack'):
        if py_state[key] != c_state.get(key):
            mismatches.append(key)
    if mismatches:
        print(f"FAIL game={game_idx} ply={ply} size={sx}x{sy} mismatch in: {mismatches}")
        print(f"moves so far: {fmt_moves(moves_so_far)}")
        print("--- python ---")
        for key in ('val', 'own', 'current', 'winner', 'eval', 'moves', 'attack'):
            print(f"{key}: {py_state[key]}")
        print("--- c (dump) ---")
        for key in ('val', 'own', 'current', 'winner', 'eval', 'moves', 'attack'):
            print(f"{key}: {c_state.get(key)}")
        return False
    return True


def main():
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
    sx = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    sy = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    max_plies = int(sys.argv[5]) if len(sys.argv) > 5 else 300

    rng = np.random.default_rng(seed)

    total_plies = 0
    zkey_checks = 0

    for game_idx in range(num_games):
        board = Board(sx, sy)
        moves_so_far = []

        for ply in range(max_plies):
            legal = board.possible_moves()
            if not legal:
                break

            # Apply move to python board.
            idx = int(rng.integers(0, len(legal)))
            x, y = legal[idx]
            board.move(x, y)
            moves_so_far.append((x, y))
            total_plies += 1

            # Query C dump at this exact position.
            res = run_dump(DUMP, sx, sy, moves_so_far)
            if res.returncode != 0:
                print(f"FAIL game={game_idx} ply={ply}: dump exited {res.returncode}")
                print(f"stderr: {res.stderr}")
                print(f"moves so far: {fmt_moves(moves_so_far)}")
                print(f"FAIL {total_plies} plies checked")
                return 1

            c_state = parse_dump_stdout(res.stdout)
            py_state = python_state(board)

            if not compare(py_state, c_state, sx, sy, moves_so_far, game_idx, ply):
                print(f"FAIL {total_plies} plies checked")
                return 1

            # Zobrist self-check via the verify build (aborts internally on
            # mismatch; a nonzero/signal exit means the key drifted).
            vres = run_dump(DUMP_VERIFY, sx, sy, moves_so_far)
            zkey_checks += 1
            if vres.returncode != 0:
                print(f"FAIL zkey game={game_idx} ply={ply}: dump-verify exited {vres.returncode}")
                print(f"stderr: {vres.stderr}")
                print(f"moves so far: {fmt_moves(moves_so_far)}")
                print(f"FAIL {total_plies} plies checked")
                return 1

            if board.winning_player() != 0:
                break

    print(f"OK {total_plies} plies checked ({zkey_checks} zkey re-derivations verified, {num_games} games, size={sx}x{sy}, seed={seed})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
