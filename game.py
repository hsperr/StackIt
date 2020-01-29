from board import Board
from alphabeta import AlphaBeta

def run_game(board):
    ai = AlphaBeta(10)
    while not board.winning_player():
        board.print()
        move = input('Please input a move! e.g.: 0,0\n>>')
        if move == 'exit':
            break
        if move == 'ai':
            move, score = ai.get_best_move(board)
            print(move, score)
            continue
        if move == 'perft':
            try:
                perft(board, AlphaBeta)
            except KeyboardInterrupt as e:
                continue

        try:
            x, y = move.split(',')
            board.move(int(x), int(y))
            move, score = ai.get_best_move(board)
            print(move, score)
            board.move(*move)
        except Exception as e:
            print(e)
    board.print()


def perft(board, AI):
    try:
        from time import time
        print("Starting performance test")
        board.print()
        print("{max_depth} - {move} - {score} - {time} - {nodes_searched} - {pv}")
        for max_depth in range(100):
            ai = AI(max_depth)
            t0 = time()
            move, score = ai.get_best_move(board)
            print(f"{max_depth} "
                  f"- {move} "
                  f"- {score} "
                  f"- {round(time()-t0, 4)} "
                  f"- {' - '.join([k + '=' + str(v) for k, v in sorted(ai.stats.items())])} "
                  f"- {ai.principle_variation}")
    except KeyboardInterrupt as e:
        pass

if __name__=='__main__':
    board_string = """
            2
            41 31 11
            11 42 31
            42 42 42
        """

    board = Board.from_string(board_string)
    run_game(board)


    #perft(board, AlphaBeta)
    #perft(board, MinMax)

    #ai = MinMax(2)
    #board.print()
    #print(ai.get_best_move(board), ai.principle_variation)
    #board.print()
