import time
import random
import math

class MonteCarloTreeSearch:

    def __init__(self):
        self.states = []
        self.max_moves = 100000

        self.plays = {}
        self.wins = {}

    def get_best_move_time(self, board, allowed_time_in_s, show_perft=False):
        print(f"MCTS with time={allowed_time_in_s} and max_moves={self.max_moves}")
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

        player = board.current_player
        board_string = board.to_string()
        next_states = [(player, board_string, move) for move in poss_moves]

        print(games)
        rates = sorted((self.wins.get(S, 0) / float(self.plays.get(S, 1)), S[-1], self.plays.get(S, 0)) for S in next_states)
        for pct_win, move, plays in rates:
            print(pct_win, move, plays)

        score, move, _ = max(rates)
        return move, score

    def _run_simulation(self, board):

        expand = True
        visits = []

        plays, wins = self.plays, self.wins

        player = board.current_player
        for m in range(self.max_moves):
            poss_moves = board.possible_moves()

            board_string = board.to_string()
            next_states = [(player, board_string, move) for move in poss_moves]
            if all(state in plays for state in next_states):
                next_moves = []
                for state in next_states:
                    _, _, move = state
                    s_i = plays[state]
                    w_i = wins[state]
                    s_p = sum(plays[s] for s in next_states)
                    r = w_i/s_i + 1.4 * math.sqrt(math.log(s_p)/s_i)
                    next_moves.append((r, move))

                _, move = sorted(next_moves)[-1]
            else:
                move = random.choice(poss_moves)

            state = (player, board.to_string(), move)
            board.move(*move)

            if expand and not state in plays:
                expand = False
                plays[state] = 0
                wins[state] = 0

            if state in plays:
                visits.append(state)
            player = board.current_player
            winner = board.winning_player()
            if winner:
                break

        if board.boxes_for(board.current_player) > board.boxes_for(board.other_player):
            winner = board.current_player
        elif board.boxes_for(board.current_player) < board.boxes_for(board.other_player):
            winner = board.other_player
        else:
            winner = 0

        for state in visits:
            plays[state] += 1
            if winner == state[0]:
                wins[state] += 1

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
    mcts.get_best_move_time(board, 60)








