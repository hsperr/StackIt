"""StackIt Arena — pit AI engines (and different code versions) against each other.

Each participant = (version, engine). A "version" is a directory containing a
StackIt checkout; an "engine" is alphabeta | mcts | minmax. Every participant
runs in its own subprocess (arena_player.py) launched with cwd = its version
dir, so baseline and improved code can play in the SAME tournament without
Python module-name clashes. This arena process is the neutral REFEREE: it owns
the canonical board, applies every move, and judges the winner — so both sides
play under one ruleset regardless of what their own board.py does.

Examples
--------
  # Round-robin among the three engines in the current checkout:
  python3 arena.py

  # Baseline vs improved AlphaBeta (create baseline worktree first):
  #   git worktree add ../stackit-baseline f6b6bd9
  python3 arena.py --versions "base=../stackit-baseline,dev=." --engines alphabeta

  # Everything vs everything, 1s/move, 4 games per pairing:
  python3 arena.py --engines alphabeta,mcts --time 1.0 --games 4
"""
import sys
import os
import json
import time
import argparse
import subprocess
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from board import Board  # referee uses THIS checkout's rules for everyone

PLAYER_SCRIPT = os.path.join(HERE, 'arena_player.py')


# ---------------------------------------------------------------- participants
class Participant:
    def __init__(self, name, version_dir, engine, minmax_depth):
        self.name = name                     # e.g. "dev/alphabeta"
        self.version_dir = os.path.abspath(version_dir)
        self.engine = engine
        self.minmax_depth = minmax_depth
        self.proc = None

    def start(self):
        env = dict(os.environ)
        env['PYTHONPATH'] = self.version_dir + os.pathsep + env.get('PYTHONPATH', '')
        self.proc = subprocess.Popen(
            [sys.executable, '-u', PLAYER_SCRIPT, self.engine, str(self.minmax_depth)],
            cwd=self.version_dir, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.name}: worker failed to start "
                               f"(is board.py present in {self.version_dir}?)")
        hello = json.loads(line)
        assert hello.get('ready'), hello

    def ask(self, board, budget, max_depth):
        req = {"cmd": "move", "board": board.board, "player": board.player,
               "current": board.current_player, "budget": budget, "max_depth": max_depth}
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.name}: worker died mid-game")
        return json.loads(line)

    def stop(self):
        if not self.proc:
            return
        try:
            self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# --------------------------------------------------------------------- openings
def empty_opening(size):
    b = Board(size, size)
    return {"name": f"empty{size}x{size}", "board": b.board,
            "player": b.player, "current": 1}


def preset_openings():
    """A couple of asymmetric mid-game positions for variety (from game.py/mcts.py)."""
    out = []
    strings = {
        "midgame-a": """1
            32 00 31 42
            11 42 32 00
            31 11 42 32
            41 31 00 41""",
    }
    for name, s in strings.items():
        b = Board.from_string(s)
        out.append({"name": name, "board": b.board, "player": b.player, "current": 1})
    return out


# ----------------------------------------------------------------- game runner
def winner_by_boxes(board):
    b1, b2 = board.boxes_for(1), board.boxes_for(2)
    return 1 if b1 > b2 else 2 if b2 > b1 else 0


def play_game(p1, p2, opening, budget, max_depth, max_moves):
    """Return (winner in {0,1,2}, reason, plies). p1 is player 1, p2 is player 2."""
    board = Board.from_custom_board(opening['board'], opening['player'], opening['current'])
    players = {1: p1, 2: p2}

    for ply in range(max_moves):
        w = board.winning_player()
        if w:
            return w, "domination", ply
        moves = board.possible_moves()
        if not moves:
            return winner_by_boxes(board), "no-moves", ply

        cur = players[board.current_player]
        mover = board.current_player
        resp = cur.ask(board, budget, max_depth)

        if not resp.get('ok') or resp.get('move') is None:
            return board.other_player, f"resign({resp.get('error', 'no-move')})", ply
        mv = tuple(resp['move'])
        if mv not in moves:
            return board.other_player, f"illegal-move{mv}", ply

        board.move(*mv)

    return winner_by_boxes(board), "move-cap", max_moves


# ---------------------------------------------------------------------- tourney
def run(args):
    versions = {}
    for chunk in args.versions.split(','):
        name, _, path = chunk.partition('=')
        versions[name.strip()] = (path.strip() or '.')
    engines = [e.strip() for e in args.engines.split(',') if e.strip()]

    participants = []
    for vname, vdir in versions.items():
        for eng in engines:
            label = f"{vname}/{eng}" if len(versions) > 1 else eng
            participants.append(Participant(label, vdir, eng, args.minmax_depth))

    if len(participants) < 2:
        print("Need at least 2 participants. Add more --engines or --versions.")
        return

    openings = [empty_opening(args.size)]
    if args.openings == 'all':
        openings += preset_openings()

    pairs = list(itertools.combinations(participants, 2))
    total_games = len(pairs) * args.games * len(openings)
    time_engines = any(e in ('alphabeta', 'mcts') for e in engines)
    est = total_games * args.max_moves * args.time if time_engines else None

    print("=" * 64)
    print(f"StackIt Arena — {len(participants)} participants, {len(pairs)} pairings")
    print(f"participants : {', '.join(p.name for p in participants)}")
    print(f"openings     : {', '.join(o['name'] for o in openings)}")
    print(f"per pairing  : {args.games} game(s) x {len(openings)} opening(s), colors swapped")
    print(f"time/move    : {args.time}s (minmax is depth-{args.minmax_depth}, not timed)")
    print(f"total games  : {total_games}")
    if est is not None:
        print(f"est. wall    : <= ~{est/60:.1f} min (upper bound; most moves finish faster)")
    print(f"wall guard   : {args.wall}s")
    print("=" * 64)

    for p in participants:
        p.start()

    # head-to-head[a][b] = wins of a over b ; standings
    stand = {p.name: {"w": 0, "l": 0, "d": 0} for p in participants}
    h2h = {p.name: {q.name: 0 for q in participants} for p in participants}

    t0 = time.time()
    game_no = 0
    aborted = False
    try:
        for a, b in pairs:
            for g in range(args.games):
                for opening in openings:
                    if time.time() - t0 > args.wall:
                        print(f"\n[wall guard hit at {args.wall}s — stopping early]")
                        aborted = True
                        break
                    # swap colors every game for balance
                    first, second = (a, b) if g % 2 == 0 else (b, a)
                    game_no += 1
                    winner, reason, plies = play_game(
                        first, second, opening, args.time, args.max_depth, args.max_moves)

                    if winner == 1:
                        wname, lname = first.name, second.name
                    elif winner == 2:
                        wname, lname = second.name, first.name
                    else:
                        wname = lname = None

                    if wname is None:
                        stand[first.name]["d"] += 1
                        stand[second.name]["d"] += 1
                        tag = "draw"
                    else:
                        stand[wname]["w"] += 1
                        stand[lname]["l"] += 1
                        h2h[wname][lname] += 1
                        tag = f"{wname} beats {lname}"

                    print(f"  [{game_no}/{total_games}] {opening['name']:>12} | "
                          f"P1={first.name} P2={second.name} -> {tag} "
                          f"({reason}, {plies} plies, {time.time()-t0:.0f}s)")
                if aborted:
                    break
            if aborted:
                break
    finally:
        for p in participants:
            p.stop()

    # ---- standings
    print("\n" + "=" * 64)
    print("STANDINGS")
    print("=" * 64)
    order = sorted(stand.items(), key=lambda kv: (kv[1]["w"] - kv[1]["l"], kv[1]["w"]), reverse=True)
    width = max(len(n) for n in stand)
    print(f"{'engine'.ljust(width)}   W   L   D   pts")
    for name, s in order:
        pts = s["w"] - s["l"]
        print(f"{name.ljust(width)}  {s['w']:>2}  {s['l']:>2}  {s['d']:>2}  {pts:>+4}")

    # ---- head-to-head grid
    print("\nHEAD-TO-HEAD (row wins vs col)")
    names = [p.name for p in participants]
    colw = max(6, max(len(n) for n in names))
    print(" " * (width + 2) + " ".join(n[:colw].rjust(colw) for n in names))
    for rn in names:
        cells = []
        for cn in names:
            cells.append(("-" if rn == cn else str(h2h[rn][cn])).rjust(colw))
        print(rn.ljust(width + 2) + " ".join(cells))
    print(f"\nElapsed: {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description="StackIt AI arena")
    ap.add_argument('--versions', default='local=.',
                    help='comma list name=dir (e.g. "base=../stackit-baseline,dev=.")')
    ap.add_argument('--engines', default='alphabeta,mcts,minmax',
                    help='comma list of alphabeta|mcts|minmax')
    ap.add_argument('--time', type=float, default=1.0, help='thinking seconds/move (timed engines)')
    ap.add_argument('--games', type=int, default=2, help='games per pairing per opening (colors swap)')
    ap.add_argument('--openings', choices=['empty', 'all'], default='empty')
    ap.add_argument('--size', type=int, default=5, help='board size for the empty opening')
    ap.add_argument('--max-moves', type=int, default=50, help='ply cap; decided by box count if hit')
    ap.add_argument('--max-depth', type=int, default=20, help='max search depth for timed engines')
    ap.add_argument('--minmax-depth', type=int, default=3, help='fixed depth for the minmax engine')
    ap.add_argument('--wall', type=int, default=600, help='hard wall-clock budget in seconds')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
