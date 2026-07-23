"""Arena worker: runs ONE engine from whatever StackIt checkout it is launched in.

Launched by arena.py as a subprocess with cwd set to a specific version's
directory, so `from board import Board` (etc.) resolves to THAT version's code.
Speaks newline-delimited JSON on stdin/stdout:

  <- {"cmd":"move","board":[[...]],"player":[[...]],"current":1,"budget":1.0,"max_depth":20}
  -> {"ok":true,"move":[x,y],"score":<num|null>}
  <- {"cmd":"quit"}

This keeps two different versions of board.py/alphabeta.py loadable at the same
time (each in its own process) which is impossible inside a single interpreter.
"""
import sys
import os
import json

# Prefer the version dir we were launched in over arena.py's own directory.
sys.path.insert(0, os.getcwd())


def build_engine(engine_name, minmax_depth):
    if engine_name == 'alphabeta':
        from alphabeta import AlphaBeta
        return AlphaBeta()
    if engine_name == 'mcts':
        from mcts import MonteCarloTreeSearch
        return MonteCarloTreeSearch()
    if engine_name == 'minmax':
        from minmax import MinMax
        return MinMax(minmax_depth)
    if engine_name == 'alphazero':
        from alphazero.engine import AlphaZero
        return AlphaZero()          # loads checkpoints/best.pt (train first)
    raise ValueError(f"unknown engine: {engine_name}")


def normalize(result):
    """Engines disagree on return shape. Return (move, score) or (None, None).

    - AlphaBeta / MCTS (normal): ((x, y), score)
    - MCTS trivial position (baseline bug): bare (x, y) or None
    - MinMax: ((x, y), score)
    """
    if result is None:
        return None, None
    if isinstance(result, (tuple, list)) and len(result) == 2:
        first, second = result
        if isinstance(first, (tuple, list)):   # ((x, y), score)
            return first, second
        return result, None                    # bare (x, y)
    return None, None


def main():
    engine_name = sys.argv[1]
    minmax_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    from board import Board

    # Engines print to stdout (e.g. MCTS banner, show_perft). Keep the JSON
    # protocol on the real stdout and silence everything else.
    proto = sys.stdout
    proto.write(json.dumps({"ready": True, "engine": engine_name,
                            "cwd": os.getcwd()}) + "\n")
    proto.flush()
    sys.stdout = open(os.devnull, 'w')

    # NB: use readline() rather than `for line in sys.stdin` — the file-iterator
    # protocol reads ahead into an internal buffer and would deadlock this
    # request/response loop (worker waits to fill its buffer, parent waits for a
    # reply that never comes).
    while True:
        line = sys.stdin.readline()
        if not line:      # EOF: parent closed the pipe
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "quit":
                break

            board = Board.from_custom_board(req["board"], req["player"], req["current"])
            engine = build_engine(engine_name, minmax_depth)  # fresh: no stale table
            budget = req.get("budget", 1.0)
            max_depth = req.get("max_depth", 20)

            if engine_name == 'minmax':
                result = engine.get_best_move(board)           # depth-bounded, ignores time
            else:
                result = engine.get_best_move(board, thinking_time=budget,
                                              max_depth=max_depth, show_perft=False)

            move, score = normalize(result)
            if move is None:
                resp = {"ok": True, "move": None, "score": None}
            else:
                resp = {"ok": True,
                        "move": [int(move[0]), int(move[1])],
                        "score": score if isinstance(score, (int, float)) else None}
        except Exception as exc:
            import traceback
            resp = {"ok": False, "error": str(exc), "trace": traceback.format_exc()}

        # NB: write via `proto`, not sys.stdout — sys.stdout is redirected to
        # devnull above to swallow engine chatter; the protocol lives on the
        # real stdout captured in `proto`.
        proto.write(json.dumps(resp) + "\n")
        proto.flush()


if __name__ == '__main__':
    main()
