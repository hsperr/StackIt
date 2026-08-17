"""StackIt web server — JSON API + a single-page front end.

The browser owns all rendering; Flask only holds game state and runs engines.
Every engine reply carries a unified `analysis` block so the UI can show the
same panels for AlphaBeta, MCTS and AlphaZero (principal variation, candidate
moves, and for AlphaZero the raw network policy/value).

    python3 server.py            # http://localhost:9999
"""
import os
import time
import uuid
import threading
from collections import OrderedDict

from flask import Flask, render_template, request, jsonify

from board import Board
from alphabeta import AlphaBeta
from mcts import MonteCarloTreeSearch
from utils import StackItException

app = Flask(__name__)


# Hard limits. These are safety rails, not tuning knobs: a visitor picks the
# board size and thinking time, so without them one request can pin a core for
# a minute or chew through the box's memory. Sized for a small server (1 GB
# RAM, 1-2 cores). Raise them if you run this somewhere bigger.
MAX_THINKING_TIME = 5     # seconds an engine may search for one move
MAX_BOARD = 8             # largest board a visitor may start
MAX_GAMES = 16            # live games kept in memory; the oldest is dropped
MAX_FRAMES = 48           # animation frames per move, oldest cascade rings kept

# One AlphaZero engine per board size, shared by every game — loading the
# checkpoint and building the evaluator is far too slow to redo per game. The
# engine carries search state, so only one request may use it at a time.
_az_engines = {}
_az_lock = threading.Lock()
_store_lock = threading.Lock()


# --------------------------------------------------------------- engines ----

ENGINES = {
    "alphabeta": {
        "name": "AlphaBeta",
        "blurb": "Classic game-tree search with pruning. Shows its principal variation.",
        "unit": "score",
    },
    "mcts": {
        "name": "Monte Carlo Tree Search",
        "blurb": "Plays thousands of random games and keeps what wins.",
        "unit": "winrate",
    },
    "alphazero": {
        "name": "AlphaZero",
        "blurb": "Neural network guiding the search. Shows what the net thinks.",
        "unit": "winprob",
    },
}


def az_engine(size):
    """Return the AlphaZero engine for `size`x`size`, or raise with a message
    the UI can show the user (no checkpoint / wrong board size / no torch).

    Callers that *search* with the returned engine must hold `_az_lock`; the
    engine keeps its MCTS tree on the instance, so two concurrent searches
    would read and overwrite each other's nodes.
    """
    if size in _az_engines:
        return _az_engines[size]
    try:
        from alphazero.engine import AlphaZero
        from alphazero.config import Config
    except ImportError as e:
        raise RuntimeError(f"AlphaZero needs PyTorch: {e}")

    ckpt = os.path.join(Config().ckpt_dir, "best.pt")
    if not os.path.exists(ckpt):
        raise RuntimeError(
            f"No trained network at {ckpt}. Train one first: python3 -m alphazero.train")
    engine = AlphaZero(ckpt=ckpt)
    engine._ensure_loaded(Board(size, size))   # raises if the net is another size
    _az_engines[size] = engine
    return engine


def az_available(size):
    try:
        az_engine(size)
        return True, None
    except Exception as e:
        return False, str(e)


# ------------------------------------------------------------------ game ----

class Game:
    def __init__(self, iid, size=5, mode="hva", ai="alphabeta",
                 human_seat=1, thinking_time=3, max_depth=30):
        self.iid = iid
        self.board = Board(size, size)
        self.mode = mode                  # "hvh" (human vs human) or "hva"
        self.ai = ai
        self.human_seat = human_seat      # which player the human controls in "hva"
        self.thinking_time = thinking_time
        self.max_depth = max_depth
        self.alphabeta = AlphaBeta()
        self.mcts = MonteCarloTreeSearch()
        self.moves = []                   # [(x, y), ...] in play order
        self.last_analysis = None
        # Serialises requests touching this game. Two clicks landing at once
        # would otherwise interleave inside board.move() and corrupt history.
        self.lock = threading.Lock()

    # -- state -------------------------------------------------------------

    def ai_seat(self):
        return 2 if self.human_seat == 1 else 1

    def is_ai_turn(self):
        return (self.mode == "hva"
                and self.board.current_player == self.ai_seat()
                and not self.status()["over"])

    def status(self):
        w = self.board.winning_player()
        if w:
            return {"over": True, "winner": w, "reason": "domination"}
        if not self.board.possible_moves():
            b1, b2 = self.board.boxes_for(1), self.board.boxes_for(2)
            return {"over": True, "reason": "no moves",
                    "winner": 1 if b1 > b2 else 2 if b2 > b1 else 0}
        return {"over": False, "winner": 0, "reason": None}

    def cells(self):
        """Flat [value, owner] pairs in row-major order."""
        out = []
        for y, row in enumerate(self.board.board):
            for x, v in enumerate(row):
                out.append([v, self.board.player[y][x]])
        return out

    def seat_labels(self):
        if self.mode == "hvh":
            return {"1": "Player 1", "2": "Player 2"}
        ai_name = ENGINES[self.ai]["name"]
        return {str(self.human_seat): "You",
                str(self.ai_seat()): ai_name}

    def state(self):
        b = self.board
        st = self.status()
        return {
            "iid": self.iid,
            "size_x": b.size_x, "size_y": b.size_y,
            "cells": self.cells(),
            "current_player": b.current_player,
            "legal": [[x, y] for (x, y) in b.possible_moves()],
            "mode": self.mode,
            "ai": self.ai,
            "ai_name": ENGINES[self.ai]["name"],
            "human_seat": self.human_seat,
            "ai_seat": self.ai_seat(),
            "thinking_time": self.thinking_time,
            "max_depth": self.max_depth,
            "labels": self.seat_labels(),
            "cell_count": {"1": sum(1 for r in b.player for v in r if v == 1),
                           "2": sum(1 for r in b.player for v in r if v == 2)},
            "blocks": {"1": b.boxes_for(1), "2": b.boxes_for(2)},
            "over": st["over"], "winner": st["winner"], "reason": st["reason"],
            "move_count": len(self.moves),
            "last_move": list(self.moves[-1]) if self.moves else None,
            "can_undo": len(self.moves) > 0,
            "is_ai_turn": self.is_ai_turn(),
        }

    # -- moving ------------------------------------------------------------

    def play(self, x, y):
        """Play a move, returning the intermediate positions of the chain
        reaction so the front end can animate the real toppling."""
        frames = []

        def snap(_board):
            frames.append(self.cells())

        self.board.move(x, y, on_step=snap)
        self.moves.append((x, y))
        if len(frames) > MAX_FRAMES:      # keep the first and last rings
            head = frames[:MAX_FRAMES - 1]
            frames = head + [frames[-1]]
        return frames

    def undo(self):
        """Undo back to the human's turn (both half-moves in human-vs-AI)."""
        steps = 1
        if self.mode == "hva" and len(self.moves) >= 2:
            steps = 2
        steps = min(steps, len(self.board.history))
        for _ in range(steps):
            self.board.undo()
            if self.moves:
                self.moves.pop()
        self.last_analysis = None


# ------------------------------------------------------------- analysis ----

def analyse_alphabeta(game):
    ab = game.alphabeta
    t0 = time.time()
    move, score = ab.get_best_move(game.board, thinking_time=game.thinking_time,
                                   max_depth=game.max_depth)
    elapsed = time.time() - t0
    ladder = ab.iter_log
    last = ladder[-1] if ladder else {}
    # A forced move short-circuits the search, so there is no score to show.
    if score is None:
        text, caption = "—", "only one move was possible"
    else:
        text = ("+" if score > 0 else "") + str(score)
        if score > 0:
            caption = "blocks ahead, it reckons"
        elif score < 0:
            caption = "blocks behind, it reckons"
        else:
            caption = "it sees the game as level"
    return {
        "engine": "alphabeta",
        "name": "AlphaBeta",
        "move": list(move) if move else None,
        "elapsed": round(elapsed, 2),
        "headline": {"kind": "score", "value": score, "text": text,
                     "caption": caption},
        "pv": last.get("pv", []),
        "depth": last.get("depth"),
        "nodes": last.get("nodes"),
        "ladder": ladder[-8:],
        "candidates": [],
        "net": None,
    }, move


def analyse_mcts(game):
    m = game.mcts
    t0 = time.time()
    move, score = m.get_best_move(game.board, thinking_time=game.thinking_time,
                                  max_depth=game.max_depth)
    elapsed = time.time() - t0
    rows = m.perft[:8]
    candidates = []
    total = 0
    for r in rows:
        plays = 0
        stat = r[4]
        if isinstance(stat, str) and "num_games=" in stat:
            try:
                plays = int(stat.split("num_games=")[1].split(",")[0])
            except (ValueError, IndexError):
                plays = 0
        total = max(total, plays)
        candidates.append({
            "move": list(r[1]),
            "value": round(float(r[2]), 3),
            "value_text": f"{round(float(r[2]) * 100)}%",
            "weight": plays,
            "pv": [list(mv) for mv in r[5][:8]],
        })
    pv = candidates[0]["pv"] if candidates else []
    wp = float(score) if score is not None else None
    return {
        "engine": "mcts",
        "name": "Monte Carlo Tree Search",
        "move": list(move) if move else None,
        "elapsed": round(elapsed, 2),
        "headline": {"kind": "winrate", "value": wp,
                     "text": f"{round(wp * 100)}%" if wp is not None else "—",
                     "caption": "games it won from here"},
        "pv": pv,
        "depth": None,
        "nodes": total,
        "ladder": [],
        "candidates": candidates,
        "net": None,
    }, move


def analyse_alphazero(game):
    engine = az_engine(game.board.size_x)
    t0 = time.time()
    with _az_lock:                      # the engine's search tree is not shareable
        info = engine.analyze(game.board, thinking_time=game.thinking_time)
    elapsed = time.time() - t0
    if info is None:
        return None, None
    move = tuple(info["move"])
    nx = game.board.size_x
    legal = set(game.board.possible_moves())
    # net_policy is a full board-sized vector; surface only the legal cells so
    # the heat map matches what the net could actually play.
    policy = []
    for i, p in enumerate(info["net_policy"]):
        cell = (i % nx, i // nx)
        if cell in legal:
            policy.append([cell[0], cell[1], round(float(p), 4)])
    candidates = [{
        "move": c["move"],
        "value": c["q"],
        "value_text": f"{round((c['q'] + 1) / 2 * 100)}%",
        "weight": c["visits"],
        "prior": c["prior"],
        "pv": [],
    } for c in info["top"]]
    return {
        "engine": "alphazero",
        "name": "AlphaZero",
        "move": list(move),
        "elapsed": round(elapsed, 2),
        "headline": {"kind": "winprob", "value": info["win_prob"],
                     "text": f"{round(info['win_prob'] * 100)}%",
                     "caption": "its chance to win after searching"},
        "pv": info["pv"],
        "depth": None,
        "nodes": info["sims"],
        "ladder": [],
        "candidates": candidates,
        "net": {
            "win_prob": info["net_win_prob"],
            "win_prob_text": f"{round(info['net_win_prob'] * 100)}%",
            "policy": policy,
        },
    }, move


ANALYSERS = {"alphabeta": analyse_alphabeta,
             "mcts": analyse_mcts,
             "alphazero": analyse_alphazero}


# ----------------------------------------------------------------- store ----

games = OrderedDict()


def get_game(iid):
    with _store_lock:
        g = games.get(iid)
        if g is not None:
            games.move_to_end(iid)
        return g


def put_game(g):
    with _store_lock:
        games[g.iid] = g
        games.move_to_end(g.iid)
        while len(games) > MAX_GAMES:
            games.popitem(last=False)


# ------------------------------------------------------------------ API -----

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/engines")
def api_engines():
    size = request.args.get("size", type=int, default=5)
    out = []
    for eid, meta in ENGINES.items():
        available, reason = (True, None)
        if eid == "alphazero":
            available, reason = az_available(size)
        out.append({"id": eid, "name": meta["name"], "blurb": meta["blurb"],
                    "available": available, "reason": reason})
    # The form reads its own limits from here, so the caps live in one place.
    return jsonify({"engines": out,
                    "limits": {"max_board": MAX_BOARD,
                               "max_thinking_time": MAX_THINKING_TIME}})


@app.route("/api/game", methods=["POST"])
def api_new_game():
    d = request.get_json(silent=True) or {}
    size = max(2, min(MAX_BOARD, int(d.get("size", 5))))
    mode = d.get("mode", "hva")
    if mode not in ("hvh", "hva"):
        mode = "hva"
    ai = d.get("ai", "alphabeta")
    if ai not in ENGINES:
        ai = "alphabeta"
    human_seat = 2 if int(d.get("human_seat", 1)) == 2 else 1
    thinking_time = max(1, min(MAX_THINKING_TIME, int(d.get("thinking_time", 3))))
    max_depth = max(1, min(100, int(d.get("max_depth", 30))))

    if mode == "hva" and ai == "alphazero":
        ok, reason = az_available(size)
        if not ok:
            return jsonify({"error": reason}), 400

    g = Game(str(uuid.uuid4()), size=size, mode=mode, ai=ai,
             human_seat=human_seat, thinking_time=thinking_time,
             max_depth=max_depth)
    put_game(g)
    return jsonify({"state": g.state()})


@app.route("/api/game/<iid>")
def api_state(iid):
    g = get_game(iid)
    if g is None:
        return jsonify({"error": "That game expired. Start a new one."}), 404
    return jsonify({"state": g.state(), "analysis": g.last_analysis})


@app.route("/api/game/<iid>/move", methods=["POST"])
def api_move(iid):
    g = get_game(iid)
    if g is None:
        return jsonify({"error": "That game expired. Start a new one."}), 404
    d = request.get_json(silent=True) or {}
    try:
        x, y = int(d["x"]), int(d["y"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Bad move."}), 400
    with g.lock:
        if (x, y) not in g.board.possible_moves():
            return jsonify({"error": "That cell is not yours to play."}), 400
        if g.is_ai_turn():
            return jsonify({"error": "It is the AI's turn."}), 400
        try:
            frames = g.play(x, y)
        except StackItException as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"state": g.state(), "frames": frames, "move": [x, y]})


@app.route("/api/game/<iid>/ai", methods=["POST"])
def api_ai_move(iid):
    g = get_game(iid)
    if g is None:
        return jsonify({"error": "That game expired. Start a new one."}), 404
    with g.lock:
        if not g.is_ai_turn():
            return jsonify({"error": "Not the AI's turn."}), 400
        try:
            analysis, move = ANALYSERS[g.ai](g)
        except Exception as e:
            return jsonify({"error": f"{ENGINES[g.ai]['name']} failed: {e}"}), 500
        if move is None:
            return jsonify({"state": g.state(), "frames": [], "analysis": None})
        frames = g.play(*move)
        g.last_analysis = analysis
        return jsonify({"state": g.state(), "frames": frames,
                        "move": list(move), "analysis": analysis})


@app.route("/api/game/<iid>/netread")
def api_net_read(iid):
    """The trained net's raw read of the position on screen right now: one
    forward pass, no search. Drives the heat overlay so what the board shows
    and what the panel says always describe the same position."""
    g = get_game(iid)
    if g is None:
        return jsonify({"error": "That game expired. Start a new one."}), 404
    if g.ai != "alphazero" or g.mode != "hva" or g.status()["over"]:
        return jsonify({"net": None})
    try:
        engine = az_engine(g.board.size_x)
        with _az_lock:
            priors, value = engine.mcts.ev.infer(g.board)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    nx = g.board.size_x
    legal = set(g.board.possible_moves())
    policy = [[i % nx, i // nx, round(float(p), 4)]
              for i, p in enumerate(priors) if (i % nx, i // nx) in legal]
    win_prob = (float(value) + 1) / 2
    return jsonify({"net": {
        "win_prob": round(win_prob, 3),
        "win_prob_text": f"{round(win_prob * 100)}%",
        "for_player": g.board.current_player,
        "policy": policy,
    }})


@app.route("/api/game/<iid>/undo", methods=["POST"])
def api_undo(iid):
    g = get_game(iid)
    if g is None:
        return jsonify({"error": "That game expired. Start a new one."}), 404
    with g.lock:
        g.undo()
        return jsonify({"state": g.state(), "analysis": None})


@app.route("/api/game/<iid>/settings", methods=["POST"])
def api_settings(iid):
    g = get_game(iid)
    if g is None:
        return jsonify({"error": "That game expired. Start a new one."}), 404
    d = request.get_json(silent=True) or {}
    if "thinking_time" in d:
        g.thinking_time = max(1, min(MAX_THINKING_TIME, int(d["thinking_time"])))
    if "max_depth" in d:
        g.max_depth = max(1, min(100, int(d["max_depth"])))
    return jsonify({"state": g.state()})


@app.route("/healthz")
def healthz():
    """Liveness probe for the service manager / reverse proxy."""
    return jsonify({"ok": True, "games": len(games)})


if __name__ == "__main__":
    # Development only. In production run it through gunicorn — see DEPLOY.md.
    app.run(host="0.0.0.0", port=9999, debug=True)
