from collections import defaultdict
import time


def rotate_move(p, x, y):
    """Coordinates of move p=(x,y) after board.rotate() (a 90 degree
    rotation via transpose+row-reverse). `x`/`y` must be the *new*
    (already-rotated) board's size_x/size_y."""
    return (x - 1 - p[1], p[0])


def flip_move(p, x, y):
    """Coordinates of move p=(x,y) after board.flip() (row order
    reversed, dimensions unchanged)."""
    return (p[0], y - 1 - p[1])


class AlphaBeta:
    EXACT_MATCH = 1
    UPPERBOUND = 2
    LOWERBOUND = 3

    def __init__(self, debug=False):
        self.debug = debug
        self.hashtable = {}
        self.perft = []

    def __repr__(self):
        return f"AlphaBeta()"

    def get_pv(self, board):
        pv = []
        cnt = 0

        while True:
            entry = self.hashtable.get(board.hash(), None)
            if entry:
                hash_depth, hash_move, hash_alpha, hash_beta, hash_type, _ = entry
                pv.append(hash_move)
                if not hash_move in board.possible_moves():
                    break
                cnt +=1
                board.move(*hash_move)
            else:
                break

        for i in range(cnt):
            board.undo()

        return pv

    def get_best_move(self, board, thinking_time=30, max_depth=1000000, show_perft=False):
        self.stats = defaultdict(int)
        self.hashtable = {}
        self.start_time = time.time()
        self.allowed_time = thinking_time
        self.perft = []

        poss_moves = board.possible_moves()
        if len(poss_moves) == 0:
            return None, None
        if len(poss_moves) == 1:
            return poss_moves[0], 0

        depth = 0
        best_move, best_score = None, None
        while not self.time_over() and depth<=max_depth:
            move, score = self._maxmimize(board, -1000000, 1000000, depth)

            if not self.time_over():
                self.perft.append([f"{depth} ",
                          f"{move} ",
                          f"{score} ",
                          f"{round(time.time()-self.start_time, 2)} ",
                          f"{' - '.join([k + '=' + str(v) for k, v in sorted(self.stats.items())])} ",
                          f"{self.get_pv(board)[:4]}",
                                  ])

            if show_perft:
                print("- ".join(self.perft[-1]))

            depth += 1
            if time.time()-self.start_time >= thinking_time:
                break

            best_move, best_score = move, score

        return best_move, best_score

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
        # Compute the canonical string once and reuse it for both the hash key
        # and the collision comparison below (was computed twice: hash() also
        # calls to_string()).
        current_string = board.to_string()
        hash_entry = self.hashtable.get(hash(current_string), None)

        if hash_entry:
            hash_depth, hash_move, hash_alpha, hash_beta, hash_type, board_string = hash_entry
            if board_string == current_string:
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

            # Reuse a single to_string() per board for both the hash key and the
            # stored value (previously hash() recomputed to_string(), doubling
            # the work for each of the 8 stored symmetries).
            self.hashtable[hash(current_string)] = (depth, best_move, alpha, beta, hash_type, current_string)

            # Also store this evaluation under the hashes of the 7 other
            # boards that are symmetric to this one (3 rotations, a flip,
            # and the flip's 3 rotations), so a transposition into any of
            # those equivalent orientations can reuse it. `best_move` must
            # be transformed into each orientation's own coordinates too -
            # storing it unrotated (or rotating some unrelated stale value)
            # would hand back a move that is invalid, or silently wrong, for
            # the board actually being searched. `bs` is computed once per
            # store (board.hash() would recompute to_string internally).
            if best_move is not None:
                sym_move = best_move

                board = board.rotate()
                sym_move = rotate_move(sym_move, board.size_x, board.size_y)
                bs = board.to_string()
                self.hashtable[hash(bs)] = (depth, sym_move, alpha, beta, hash_type, bs)

                board = board.rotate()
                sym_move = rotate_move(sym_move, board.size_x, board.size_y)
                bs = board.to_string()
                self.hashtable[hash(bs)] = (depth, sym_move, alpha, beta, hash_type, bs)

                board = board.rotate()
                sym_move = rotate_move(sym_move, board.size_x, board.size_y)
                bs = board.to_string()
                self.hashtable[hash(bs)] = (depth, sym_move, alpha, beta, hash_type, bs)

                board = board.flip()
                sym_move = flip_move(sym_move, board.size_x, board.size_y)
                bs = board.to_string()
                self.hashtable[hash(bs)] = (depth, sym_move, alpha, beta, hash_type, bs)

                board = board.rotate()
                sym_move = rotate_move(sym_move, board.size_x, board.size_y)
                bs = board.to_string()
                self.hashtable[hash(bs)] = (depth, sym_move, alpha, beta, hash_type, bs)

                board = board.rotate()
                sym_move = rotate_move(sym_move, board.size_x, board.size_y)
                bs = board.to_string()
                self.hashtable[hash(bs)] = (depth, sym_move, alpha, beta, hash_type, bs)

                board = board.rotate()
                sym_move = rotate_move(sym_move, board.size_x, board.size_y)
                bs = board.to_string()
                self.hashtable[hash(bs)] = (depth, sym_move, alpha, beta, hash_type, bs)


        return best_move, best_score
