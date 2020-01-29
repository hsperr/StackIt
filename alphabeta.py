from collections import defaultdict
import time

class AlphaBeta:
    EXACT_MATCH = 1
    UPPERBOUND = 2
    LOWERBOUND = 3

    def __init__(self, debug=False):
        self.debug = debug
        self.hashtable = {}

    def get_pv(self, board):
        pv = []
        cnt = 0

        while True:
            entry = self.hashtable.get(board.hash(), None)
            if entry:
                cnt +=1
                hash_depth, hash_move, hash_alpha, hash_beta, hash_type = entry
                pv.append(hash_move)
                board.move(*hash_move)
            else:
                break

        for i in range(cnt):
            board.undo()

        return pv

    def get_best_move_time(self, board, allowed_time_in_s, show_perft=False):
        self.stats = defaultdict(int)
        self.hashtable = {}
        self.start_time = time.time()
        self.allowed_time = allowed_time_in_s

        depth = 0
        best_move, best_score = None, None
        while not self.time_over():
            move, score = self._maxmimize(board, -1000000, 1000000, depth)
            depth += 1
            if show_perft:
                print(f"{depth} "
                      f"- {move} "
                      f"- {score} "
                      f"- {round(time.time()-self.start_time, 2)} "
                      f"- {' - '.join([k + '=' + str(v) for k, v in sorted(self.stats.items())])} "
                      f"- {self.get_pv(board)[:4]}")
            if time.time()-self.start_time >= allowed_time_in_s:
                break

            best_move, best_score = move, score

        return best_move, best_score

    def get_best_move_depth(self, board, depth, show_perft=False):
        self.stats = defaultdict(int)
        self.hashtable = {}
        self.start_time = time.time()
        self.allowed_time = None

        for i in range(depth+1):
            move, score = self._maxmimize(board, -1000000, 1000000, i)
            if show_perft:
                print(f"{i} "
                      f"- {move} "
                      f"- {score} "
                      f"- {round(time.time()-self.start_time, 2)} "
                      f"- {' - '.join([k + '=' + str(v) for k, v in sorted(self.stats.items())])} "
                      f"- {self.get_pv(board)}")

        return move, score

    def time_over(self):
        return self.allowed_time and time.time()-self.start_time>self.allowed_time

    def _maxmimize(self, board, alpha, beta, depth):
        if self.time_over():
            return None, -1000000

        original_alpha = alpha

        if depth == 0:
            score = board.boxes_for(board.current_player) - board.boxes_for(board.other_player)
            return None, score

        if board.winning_player():
            return None, -1000000

        best_score = -10000000
        best_move = None

        hash_move = None
        hash_entry = self.hashtable.get(board.hash(), None)
        if hash_entry:
            hash_depth, hash_move, hash_alpha, hash_beta, hash_type = hash_entry
            if hash_depth >= depth:
                if hash_type == AlphaBeta.EXACT_MATCH:
                    self.stats['hash_exact'] += 1
                    return hash_move, hash_alpha
                elif hash_type == AlphaBeta.LOWERBOUND:
                    alpha = max(alpha, hash_alpha)
                elif hash_type == AlphaBeta.UPPERBOUND:
                    beta = min(beta, hash_alpha)

                if alpha >= beta:
                    self.stats['hash_cutoff'] += 1
                    return hash_move, hash_alpha

        poss_moves = board.possible_moves()
        if hash_move and hash_move in poss_moves:
            self.stats['using_hash_move_first'] += 1
            poss_moves = [hash_move] + [x for x in poss_moves if not x == hash_move]

        for move in poss_moves:
            board.move(*move)
            self.stats['moves_made'] += 1
            _, score = self._maxmimize(board, -beta, -alpha, depth - 1)
            if self.time_over():
                board.undo()
                break

            if self.debug:
                print("move", move, "score", score, 'depth', depth, 'current_player', board.current_player)
                _ = input("")

            score *= -1
            board.undo()
            if score > best_score:
                best_score = score
                best_move = move

                if self.debug:
                    print("******new best", move, score, 'depth', depth, 'current_player',
                          board.current_player)

            alpha = max(alpha, score)
            if alpha > beta:
                self.stats['beta_cutoff'] += 1
                break

        if not self.time_over():
            if best_score <= original_alpha:
                hash_type = AlphaBeta.UPPERBOUND
            elif best_score >= beta:
                hash_type = AlphaBeta.LOWERBOUND
            else:
                hash_type = AlphaBeta.EXACT_MATCH

            self.hashtable[board.hash()] = (depth, best_move, alpha, beta, hash_type)
        return best_move, best_score
