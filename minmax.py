from copy import deepcopy

class MinMax:
    def __init__(self, depth):
        self.depth = depth

    def get_best_move(self, board):
        res =  self._maxmimize(board, self.depth)
        return res

    def _maxmimize(self, board, depth):
        if depth == 0:
            score = board.boxes_for(board.current_player) - board.boxes_for(board.other_player)
            return None, score
        if  board.winning_player():
            return None, -1000000
    
        best_score = -10000000
        best_move = None
        for move in board.possible_moves():
            board.move(*move)
            _, score = self._maxmimize(board, depth-1)
            score*=-1
            board.undo()
            if score > best_score:
                best_score = score
                best_move = move
        return best_move, best_score
