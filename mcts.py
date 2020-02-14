import time
import random

class MonteCarloTreeSearch:

    def __init__(self):
        self.states = []
        self.max_moves = 50

        self.plays = {}
        self.wins = {}

    def get_best_move_time(self, board, allowed_time_in_s, show_perft=False):
        poss_moves = board.possible_moves()
        if not poss_moves:
            return
        if len(poss_moves) == 1:
            return poss_moves[0]

        start_time = time.time()
        games = 0
        while time.time() - start_time < allowed_time_in_s:
            self._run_simulation(board.copy())
            games += 1

        next_states = []
        player = board.current_player
        for move in poss_moves:
            board.move(*move)
            next_states.append((board.to_string(), move))
            board.undo()

        rates = sorted((self.wins.get((player, S), 0) / float(self.plays.get((player, S), 0)), m, self.plays.get((player, S), 0)) for S, m in next_states)
        print(games)
        for pct_win, move, plays in rates:
            print(pct_win, move, plays)

        _, move, _ = max(rates)
        return move

    def _run_simulation(self, board):

        expand = True
        visits = []

        player = board.current_player
        for m in range(self.max_moves):
            poss_moves = board.possible_moves()
            move = random.choice(poss_moves)
            board.move(*move)


            state = (player, board.to_string())
            if expand and not state in self.plays:
                expand = False
                self.plays[state] = 0
                self.wins[state] = 0


            if state in self.plays:
                visits.append(state)
            player = board.current_player
            winner = board.winning_player()
            if winner:
                break


        if board.boxes_for(board.current_player)>board.boxes_for(board.other_player):
            winner = board.current_player
        elif board.boxes_for(board.current_player)<board.boxes_for(board.other_player):
            winner = board.other_player
        else:
            winner = 0



        for state in visits:
            self.plays[state] += 1
            if winner == state[0]:
                self.wins[state] += 1

if __name__=='__main__':
    from board import Board

    board_string = """
        1
        32 00 31 42
        11 42 32 00
        31 11 42 32
        41 31 00 41
    """
    board = Board.from_string(board_string)
    board.print()

    mcts = MonteCarloTreeSearch()
    mcts.get_best_move_time(board, 120)








