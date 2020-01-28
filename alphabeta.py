from copy import deepcopy
from collections import defaultdict


class AlphaBeta:
    def __init__(self, max_depth, debug=False):
        self.max_depth = max_depth
        self.stats = defaultdict(int)
        self.debug = debug
        self.principle_variation = None

    def get_best_move(self, board):
        move, score, pv = self._maxmimize(board, -1000000, 1000000, self.max_depth)
        self.principle_variation = pv
        return move, score

    def _maxmimize(self, board, alpha, beta, depth):
        if depth == 0:
            score = board.boxes_for(board.current_player) - board.boxes_for(board.other_player)
            return None, score, []
        if board.winning_player():
            return None, -1000000, []

        best_score = -10000000
        best_move = None
        local_pv = []

        for move in board.possible_moves():
            board.move(*move)
            self.stats['moves_made'] += 1
            _, score, local_pv = self._maxmimize(board, -beta, -alpha, depth - 1)

            if self.debug:
                print("move", move, "score", score, 'depth', depth, 'current_player', board.current_player)
                _ = input("")

            score *= -1
            board.undo()
            if score > best_score:
                principle_variation = [move] + local_pv
                best_score = score
                best_move = move

                if self.debug:
                    print("******new best", move, score, principle_variation, 'depth', depth, 'current_player',
                          board.current_player)

            alpha = max(alpha, score)
            if alpha > beta:
                break

        return best_move, best_score, principle_variation
