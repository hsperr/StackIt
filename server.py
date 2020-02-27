from flask import Flask, render_template, request, redirect
from board import Board
from alphabeta import AlphaBeta
from mcts import MonteCarloTreeSearch

app = Flask(__name__)

games = {}


class Game:
    ALPHA_BETA = 'alpha'
    MCTS = 'mcts'

    COLORS = ['black', 'green', 'red']

    def __init__(self, iid: str, board: Board, ai: str, thinking_time: int):
        self.iid = iid
        self.board = board
        self.alphaBeta = AlphaBeta()
        self.mcts = MonteCarloTreeSearch()
        self.player_names = ["HUMAN", str(ai)]
        self.set_ai(ai)
        self.thinking_time = thinking_time
        self.max_depth = 30
        self.last_move = []
        self.last_move_score = []

    def set_ai(self, ai):
        self.ai = ai
        if ai == Game.ALPHA_BETA:
            self.player_names[-1] = str(self.alphaBeta)
        else:
            self.player_names[-1] = str(self.mcts)

    def is_mcts(self):
        return 'selected' if not self.is_alpha() else ''

    def is_alpha(self):
        return 'selected' if self.ai == Game.ALPHA_BETA else ''

    def get_best_move(self):
        if self.ai == Game.ALPHA_BETA:
            move, score = self.alphaBeta.get_best_move(self.board, thinking_time=self.thinking_time, max_depth=self.max_depth)
        else:
            move, score = self.mcts.get_best_move(self.board, thinking_time=self.thinking_time, max_depth=self.max_depth)

        return move, score


    def current_player(self):
        return self.player_names[self.board.current_player - 1]

    def current_player_color(self):
        return Game.COLORS[self.board.current_player]

    def board_to_template(self):
        display = []
        for y, row in enumerate(self.board.board):
            display_row = []#f"{y}-x", Game.COLORS[0], 'left', y]
            for x, field in enumerate(row):
                display_row.append(
                    (
                        f"{x}-{y}",
                        Game.COLORS[self.board.player[y][x]],
                        'left' if x == 0 else '',
                        field
                    )
                )
            display.append(display_row)
        return display

    def move(self, x, y, score=0):
       self.last_move.append((x, y))
       self.last_move_score.append(score)
       return self.board.move(x, y)

    def undo(self):
        self.board.undo()
        self.board.undo()

        self.last_move_score.pop()
        self.last_move_score.pop()

        self.last_move.pop()
        self.last_move.pop()

    def get_last_move(self):
        return self.last_move[-1] if len(self.last_move)>0 else ''

    def get_last_move_score(self):
        return self.last_move_score[-1] if len(self.last_move_score)>0 else ''

    def perft(self):
        if self.ai == Game.ALPHA_BETA:
            return self.alphaBeta.perft
        else:
            return self.mcts.perft

@app.route('/')
def index():
    import uuid
    iid = str(uuid.uuid1())
    games[iid] = Game(iid, Board(), Game.ALPHA_BETA, 20)
    return redirect(f"/game/{iid}")

@app.route('/game/<iid>')
def main(iid):
    if not iid in games:
        games[iid] = Game(iid, Board(), Game.ALPHA_BETA, 20)
    game = games[iid]
    return render_template('index.html', game=game)

@app.route('/game/<iid>/new', methods=["POST"])
def new(iid):
    sizex = int(request.form.get('x'))
    sizey = int(request.form.get('y'))
    game = games[iid]
    game.board = Board(sizex, sizey)
    return render_template('board.html', game=game)


@app.route("/game/<iid>/move", methods=["POST", "GET"])
def move(iid):
    game = games[iid]
    if request.method == "POST":
        move = [int(x) for x in request.form.get('move').split('-')]
        game.move(*move)
        return render_template('board.html', game=game)
    else:
        move, score = game.get_best_move()
        game.move(*move)
        return render_template('board.html', game=game)

@app.route("/game/<iid>/undo", methods=["POST"])
def undo(iid):
    game = games[iid]
    game.undo()
    return render_template('board.html', game=game)

@app.route("/game/<iid>/set", methods=["POST"])
def set(iid):
    game = games[iid]
    thinking_time =  int(request.form.get('thinking_time'))
    game.thinking_time = thinking_time

    max_depth =  int(request.form.get('max_depth'))
    game.max_depth = max_depth
    ai = str(request.form.get('ai'))
    game.set_ai(ai)
    return render_template('board.html', game=game)


if __name__=='__main__':
    app.run(host="0.0.0.0", port=9999, debug=True)