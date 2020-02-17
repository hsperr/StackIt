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
                hash_depth, hash_move, hash_alpha, hash_beta, hash_type, _ = entry
                pv.append(hash_move)
                if not hash_move in board.possible_moves():
                    break
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
            if show_perft:
                print(f"{depth} "
                      f"- {move} "
                      f"- {score} "
                      f"- {round(time.time()-self.start_time, 2)} "
                      f"- {' - '.join([k + '=' + str(v) for k, v in sorted(self.stats.items())])} "
                      f"- {self.get_pv(board)[:4]}")
            depth += 2
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


    def quiescense(self, board, depth):
        poss_moves = board.possible_attack_moves()
        # print(depth, poss_moves)
        if not poss_moves or depth == 0:
            score = board.boxes_for(board.current_player) - board.boxes_for(board.other_player)
            return score
        else:
            max_score = -100000000

            excludes = set()
            for move in poss_moves:
                if move in excludes:
                    continue

                self.stats['quiet_moves'] += 1

                excludes.add((move[0]+1, move[1]))
                excludes.add((move[0]-1, move[1]))
                excludes.add((move[0], move[1]+1))
                excludes.add((move[0], move[1]-1))

                # print(depth, move)
                # board.print()
                # input()
                board.move(*move)

                score = self.quiescense(board, depth - 1)
                board.undo()
                # print(depth, move, score)
                max_score = max(score, max_score)

            return max_score

    def _maxmimize(self, board, alpha, beta, depth):
        if self.time_over():
            return None, -1000000

        original_alpha = alpha

        # if depth == 0:
        #     # score = board.boxes_for(board.current_player) - board.boxes_for(board.other_player)
        #     score = self.quiescense(board, depth=100)
        #     return None, score

        if board.winning_player():
            return None, -1000000

        best_score = -10000000
        best_move = None

        hash_move = None
        hash_entry = None
        hash_entry = self.hashtable.get(board.hash(), None)

        if hash_entry:
            hash_depth, hash_move, hash_alpha, hash_beta, hash_type, board_string = hash_entry
            if board_string == board.to_string():
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
            else:
                hash_move = None
                self.stats['hash_collision'] += 1

        if depth <= -6:
            score = board.boxes_for(board.current_player) - board.boxes_for(board.other_player)
            return None, score
        elif depth <= 0:
            poss_moves = board.possible_attack_moves()
        else:
            poss_moves = board.possible_moves()

        if not poss_moves:
            score = board.boxes_for(board.current_player) - board.boxes_for(board.other_player)
            return None, score

        if hash_move and hash_move in poss_moves:
            self.stats['using_hash_move_first'] += 1
            poss_moves = [hash_move] + [x for x in poss_moves if not x == hash_move]

        for move in poss_moves:
            board.move(*move)
            if depth<=0:
                self.stats['quiet_moves'] += 1
            else:
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

            self.hashtable[board.hash()] = (depth, best_move, alpha, beta, hash_type, board.to_string())

            def rotate(p, x, y):
                if p[0] > x // 2 and p[1] > y // 2:
                    return [x - p[0] - 1, p[1]]
                if p[0] <= x // 2 and p[1] > y // 2:
                    return [p[0], y - p[1] - 1]
                if p[0] > x // 2 and p[1] <= y // 2:
                    return [p[0], y - p[1] - 1]
                if p[0] <= x // 2 and p[1] <= y // 2:
                    return [x - p[0] - 1, p[1]]

            board = board.rotate()
            move = rotate(move, board.size_x, board.size_y)
            self.hashtable[board.hash()] = (depth, best_move, alpha, beta, hash_type, board.to_string())

            board = board.rotate()
            move = rotate(move, board.size_x, board.size_y)
            self.hashtable[board.hash()] = (depth, best_move, alpha, beta, hash_type, board.to_string())

            board = board.rotate()
            move = rotate(move, board.size_x, board.size_y)
            self.hashtable[board.hash()] = (depth, best_move, alpha, beta, hash_type, board.to_string())

            board = board.flip()
            move = rotate(move, board.size_x, board.size_y)
            move = rotate(move, board.size_x, board.size_y)
            self.hashtable[board.hash()] = (depth, best_move, alpha, beta, hash_type, board.to_string())

            board = board.rotate()
            move = rotate(move, board.size_x, board.size_y)
            self.hashtable[board.hash()] = (depth, best_move, alpha, beta, hash_type, board.to_string())

            board = board.rotate()
            move = rotate(move, board.size_x, board.size_y)
            self.hashtable[board.hash()] = (depth, best_move, alpha, beta, hash_type, board.to_string())

            board = board.rotate()
            move = rotate(move, board.size_x, board.size_y)
            self.hashtable[board.hash()] = (depth, best_move, alpha, beta, hash_type, board.to_string())


        return best_move, best_score
