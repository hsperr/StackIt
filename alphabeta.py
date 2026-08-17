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

    def __init__(self, debug=False, pvs=True):
        self.debug = debug
        self.pvs = pvs             # principal-variation (null-window) search
        self.hashtable = {}
        self.perft = []

    def __repr__(self):
        return f"AlphaBeta()"

    def get_pv(self, board):
        pv = []
        cnt = 0

        while True:
            entry = self.hashtable.get(board.zkey, None)
            if entry:
                hash_depth, hash_move, hash_alpha, hash_beta, hash_type = entry
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
        self.killers = {}          # ply -> up to 2 quiet moves that caused a cutoff
        self.history_heur = {}     # move -> cumulative cutoff weight
        self.start_time = time.time()
        self.allowed_time = thinking_time
        self.perft = []
        # Same iterative-deepening rows as `perft`, but as data rather than
        # pre-formatted strings, so the web UI can render the depth ladder and
        # the principal variation without parsing display text.
        self.iter_log = []

        poss_moves = board.possible_moves()
        if len(poss_moves) == 0:
            return None, None
        if len(poss_moves) == 1:
            return poss_moves[0], 0

        depth = 0
        best_move, best_score = None, None
        while not self.time_over() and depth<=max_depth:
            self.root_depth = depth    # ply = root_depth - depth, for killer indexing
            move, score = self._maxmimize(board, -1000000, 1000000, depth)

            if not self.time_over():
                self.iter_log.append({
                    "depth": depth,
                    "move": list(move) if move else None,
                    "score": score,
                    "elapsed": round(time.time() - self.start_time, 2),
                    "nodes": self.stats.get('moves_made', 0),
                    "pv": [list(m) for m in self.get_pv(board)[:8]],
                })
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
        # The board keeps an incremental 64-bit Zobrist key, so the TT key is a
        # plain int lookup - no per-node board serialization. A full int is used
        # as the dict key (Python doesn't truncate it), so distinct positions
        # collide only on a true Zobrist collision, which is negligible at
        # search scale; we therefore skip the old string re-verification.
        key = board.zkey
        hash_entry = self.hashtable.get(key, None)

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

        # --- Move ordering: hash move, then killer moves for this ply, then the
        # history heuristic (moves that produced cutoffs elsewhere), then the
        # board's own positional pre-sort as a stable tiebreak. Good ordering is
        # what makes the null-window (PVS) searches below pay off. ---
        ply = self.root_depth - depth
        killers = self.killers.get(ply)
        k0 = killers[0] if killers else None
        k1 = killers[1] if killers and len(killers) > 1 else None
        history = self.history_heur

        if hash_move is not None and hash_move in poss_moves:
            self.stats['using_hash_move_first'] += 1

        def rank(m):
            if m == hash_move:
                return 1 << 40
            r = history.get(m, 0)
            if m == k0:
                r += 1 << 30
            elif m == k1:
                r += 1 << 29
            return r

        # Stable sort keeps the positional pre-sort order for equal-rank moves.
        poss_moves.sort(key=rank, reverse=True)

        first = True
        for move in poss_moves:
            board.move(*move)
            if depth <= 0:
                self.stats['quiet_moves'] += 1
            else:
                self.stats['moves_made'] += 1

            if first or not self.pvs:
                _, score = self._maxmimize(board, -beta, -alpha, depth - 1)
                score = -score
            else:
                # Null-window "scout": assume this move is worse than the best so
                # far and prove it cheaply. Only if it beats alpha (and matters)
                # do we pay for a full re-search.
                _, score = self._maxmimize(board, -alpha - 1, -alpha, depth - 1)
                score = -score
                if not self.time_over() and alpha < score < beta:
                    self.stats['pvs_research'] += 1
                    _, score = self._maxmimize(board, -beta, -alpha, depth - 1)
                    score = -score

            if self.time_over():
                board.undo()
                break

            board.undo()
            first = False

            if self.debug:
                print("move", move, "score", score, 'depth', depth, 'current_player', board.current_player)
                _ = input("")

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score
            if alpha >= beta:
                self.stats['beta_cutoff'] += 1
                # This move refuted the position: remember it as a killer for
                # this ply and bump its history score (deeper cutoffs weigh more).
                if k0 != move:
                    self.killers[ply] = [move] if k0 is None else [move, k0]
                history[move] = history.get(move, 0) + (depth + 7) * (depth + 7)
                break

        if not self.time_over():
            if best_score <= original_alpha:
                hash_type = AlphaBeta.UPPERBOUND
            elif best_score >= beta:
                hash_type = AlphaBeta.LOWERBOUND
            else:
                hash_type = AlphaBeta.EXACT_MATCH

            # Store under this position's Zobrist key. The 8-fold symmetry
            # store used to live here (rotations + flip), but A/B benchmarking
            # showed it net-negative: rebuilding 7 symmetric boards + hashing
            # them per node cost more than the transposition hits it bought
            # (~2.5x fewer nodes searched in the same budget). One store wins.
            self.hashtable[key] = (depth, best_move, alpha, beta, hash_type)

        return best_move, best_score
