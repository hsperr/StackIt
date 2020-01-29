from copy import deepcopy
from collections import defaultdict
import random


class AlphaBeta:
    EXACT_MATCH = 1
    UPPERBOUND = 2
    LOWERBOUND = 3

    def __init__(self, max_depth, debug=False):
        self.max_depth = max_depth
        self.debug = debug
        self.principle_variation = None
        self.hashtable = {}

    def get_best_move(self, board):
        self.stats = defaultdict(int)
        move, score, pv = self._maxmimize(board, -1000000, 1000000, self.max_depth)
        self.principle_variation = pv
        self.stats['hash_table_size'] = len(self.hashtable)
        return move, score

    def _maxmimize(self, board, alpha, beta, depth):
        original_alpha = alpha

        if depth == 0:
            score = board.boxes_for(board.current_player) - board.boxes_for(board.other_player)
            return None, score, []
        if board.winning_player():
            return None, -1000000, []

        best_score = -10000000
        best_move = None

        hash_move = None
        hash_entry = self.hashtable.get(board.hash(), None)
        if hash_entry:
            hash_depth, hash_move, hash_alpha, hash_beta, hash_type = hash_entry
            if hash_depth >= depth:
                if hash_type == AlphaBeta.EXACT_MATCH:
                    self.stats['hash_exact'] += 1
                    return hash_move, hash_alpha, [hash_move]
                elif hash_type == AlphaBeta.LOWERBOUND:
                    alpha = max(alpha, hash_alpha)
                elif hash_type == AlphaBeta.UPPERBOUND:
                    beta = min(beta, hash_alpha)

                if alpha >= beta:
                    self.stats['hash_cutoff'] += 1
                    return hash_move, hash_alpha, [hash_move]

        poss_moves = board.possible_moves()
        if hash_move and hash_move in poss_moves:
            self.stats['using_hash_move_first'] += 1
            poss_moves = [hash_move] + [x for x in poss_moves if not x == hash_move]

        for move in poss_moves:
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
                self.stats['beta_cutoff'] += 1
                break

        if best_score <= original_alpha:
            hash_type = AlphaBeta.UPPERBOUND
        elif best_score >= beta:
            hash_type = AlphaBeta.LOWERBOUND
        else:
            hash_type = AlphaBeta.EXACT_MATCH

        self.hashtable[board.hash()] = (depth, best_move, alpha, beta, hash_type)
        return best_move, best_score, principle_variation
