from flask import Flask, render_template, request
import board as b
from alphabeta import AlphaBeta

app = Flask(__name__)

games = {}
ai = AlphaBeta()


def board_to_template(board):
    display = []
    for y, row in enumerate(board.board):
        display_row = []
        for x, field in enumerate(row):
            display_row.append(
                (
                    f"{x}-{y}",
                    'red' if board.player[y][x] == 2 else 'green' if board.player[y][x] == 1 else 'black',
                    'left' if x == 0 else '',
                    field
                )
            )
        display.append(display_row)
    return display


@app.route('/')
def index():
    board_string = """
            1
            32 00 31 42
            11 42 32 00
            31 11 42 32
            41 31 00 41
        """
    board = b.Board.from_string(board_string)

    games[0] = board

    return render_template('index.html', board=board_to_template(board))

@app.route('/new', methods=["POST"])
def new():
    sizex = int(request.form.get('x'))
    sizey = int(request.form.get('y'))
    games[0] = b.Board(sizex, sizey)
    return render_template('board.html', board=board_to_template(games[0]))


@app.route("/move", methods=["POST", "GET"])
def move():
    board = games[0]
    if request.method == "POST":
        move = [int(x) for x in request.form.get('move').split('-')]
        board.move(*move)
        return render_template('board.html', board=board_to_template(board))
    else:
        move, score = ai.get_best_move_time(board, 10, show_perft=True)
        board.move(*move)
        return render_template('board.html', board=board_to_template(board))

@app.route("/undo", methods=["POST"])
def undo():
    board = games[0]
    board.undo()
    return render_template('board.html', board=board_to_template(board))


if __name__=='__main__':
    app.run(port=9999, debug=True)