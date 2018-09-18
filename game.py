from board import Board
from minmax import MinMax
from alphabeta import AlphaBeta

def run_game(board):
    ai = AlphaBeta(8)
    while not board.winning_player():
        board.print()
        move = input('Please input a move! e.g.: 0,0\n>>')
        if move == 'exit':
            break

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
        for max_depth in range(10):
            ai = AI(max_depth)
            t0 = time()
            move, score = ai.get_best_move(board)
            print(f"{max_depth} - {move} - {score} - {round(time()-t0, 4)} - {ai.stats['moves_made']} - {ai.principle_variation}")
    except KeyboardInterrupt as e:
        pass

if __name__=='__main__':
    board  = [
            [0 , 0, 0],
            [0 , 0, 0],
            [0 , 0, 0]
    ]
    player  = [
            [0 , 0, 0],
            [0 , 0, 0],
            [0 , 0, 0]
    ]
    board = Board.from_custom_board(board, player, 1)
    run_game(board)


    #perft(board, AlphaBeta)
    #perft(board, MinMax)

    #ai = MinMax(2)
    #board.print()
    #print(ai.get_best_move(board), ai.principle_variation)
    #board.print()
