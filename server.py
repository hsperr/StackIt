from flask import Flask, render_template, request, redirect
from board import Board
from alphabeta import AlphaBeta

app = Flask(__name__)

games = {}

class Game:

    COLORS = ['black', 'green', 'red']
    def __init__(self, iid: str, board: Board, ai: AlphaBeta, thinking_time: int):
        self.iid = iid
        self.board = board
        self.ai = ai
        self.thinking_time = thinking_time

    def current_player(self):
        return Game.COLORS[self.board.current_player]

    def board_to_template(self):
        display = []
        for y, row in enumerate(self.board.board):
            display_row = []
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

@app.route('/')
def index():
    import uuid
    iid = str(uuid.uuid1())
    games[iid] = Game(iid, Board(), AlphaBeta(), 20)
    return redirect(f"/game/{iid}")

@app.route('/game/<iid>')
def main(iid):
    if not iid in games:
        games[iid] = Game(iid, Board(), AlphaBeta(), 20)
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
        game.board.move(*move)
        return render_template('board.html', game=game)
    else:
        move, score = game.ai.get_best_move_time(game.board, game.thinking_time, show_perft=True)
        game.board.move(*move)
        return render_template('board.html', game=game)

@app.route("/game/<iid>/undo", methods=["POST"])
def undo(iid):
    game = games[iid]
    game.board.undo()
    return render_template('board.html', game=game)

@app.route("/game/<iid>/set", methods=["POST"])
def set(iid):
    game = games[iid]
    thinking_time =  int(request.form.get('thinking_time'))
    game.thinking_time = thinking_time
    return render_template('board.html', game=game)


if __name__=='__main__':
    app.run(host="0.0.0.0", port=9999, debug=True)