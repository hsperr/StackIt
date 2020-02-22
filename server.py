from flask import Flask, render_template, request, redirect
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
    import uuid
    iid = str(uuid.uuid1())

    board = b.Board()
    games[iid] = board
    return redirect(f"/game/{iid}")

@app.route('/game/<iid>')
def main(iid):
    if not iid in games:
        board = b.Board()
        games[iid] = board
    board = games[iid]
    return render_template('index.html', board=board_to_template(board), gameid=iid)

@app.route('/game/<iid>/new', methods=["POST"])
def new(iid):
    sizex = int(request.form.get('x'))
    sizey = int(request.form.get('y'))
    games[iid] = b.Board(sizex, sizey)
    return render_template('board.html', board=board_to_template(games[iid]), gameid=iid)


@app.route("/game/<iid>/move", methods=["POST", "GET"])
def move(iid):
    board = games[iid]
    if request.method == "POST":
        move = [int(x) for x in request.form.get('move').split('-')]
        board.move(*move)
        return render_template('board.html', board=board_to_template(board))
    else:
        move, score = ai.get_best_move_time(board, 10, show_perft=True)
        board.move(*move)
        return render_template('board.html', board=board_to_template(board), gameid=iid)

@app.route("/game/<iid>/undo", methods=["POST"])
def undo(iid):
    board = games[iid]
    board.undo()
    return render_template('board.html', board=board_to_template(board), gameid=iid)


if __name__=='__main__':
    app.run(host="0.0.0.0", port=9999, debug=True)