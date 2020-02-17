from board import Board
from alphabeta import AlphaBeta
from mcts import MonteCarloTreeSearch

def run_game():
    THINKING_TIME = 10
    HUMAN = 'human'

    PLAYERS = {1: HUMAN, 2: AlphaBeta()}

    board_string = """
            1
            32 00 31 42
            11 42 32 00
            31 11 42 32
            41 31 00 41
        """
    board = Board.from_string(board_string)

    command = ''
    while True:
        for p, t in PLAYERS.items():
            print(f"Player {p}: {t}")

        board.print()

        if PLAYERS[board.current_player] == HUMAN:
            command = input('Please input a move! e.g.: 0,0\n>>').strip()

        if command == 'exit':
            break
        elif command.startswith('new'):
            if command == 'new':
                board = Board()
            else:
                cmd, size_x, size_y = command.split(' ')
                board = Board(int(size_x), int(size_y))
        elif command == 'search':
            move, score = AlphaBeta().get_best_move_depth(board, THINKING_TIME, show_perft=True)
        elif command.startswith('set'):
            if command == 'set':
                print("Usage: set <player_number> <player_type>")
                print(f"player_type: {HUMAN}, alpha, mcts")
            else:
                cmd, player, player_type = command.split(' ')
                if player_type == HUMAN:
                    PLAYERS[int(player)] = HUMAN
                elif player_type == 'alpha':
                    PLAYERS[int(player)] = AlphaBeta()
                elif player_type == 'mcts':
                    PLAYERS[int(player)] = MonteCarloTreeSearch()
        else:
            if PLAYERS[board.current_player] == HUMAN:
                split_move = command.split(',')
                board.move(*split_move)
            else:
                move, score = PLAYERS[board.current_player].get_best_move_time(board, THINKING_TIME, show_perft=True)
                board.move(*move)

        command = ''


if __name__=='__main__':
    run_game()
