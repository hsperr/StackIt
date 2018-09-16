from copy import deepcopy


class Board:
    PLAYER1COLOR = '\033[92m'
    PLAYER2COLOR = '\033[91m'
    ENDC = '\033[0m'
    
    @classmethod
    def from_custom_board(cls, board, player, current_player=1):
        instance = cls()
        instance.board = board
        instance.player = player
        instance.current_player = current_player
        return instance

    def __init__(self):
        self.board = [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
        ]
        self.player = [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
        ]

        self.current_player = 1

        self.history = []

    @property
    def other_player(self):
        if self.current_player == 1:
            return 2
        else:
            return 1

    def possible_moves(self):
        moves = []
        for y, row in enumerate(self.board):
            for x, field in enumerate(row):
                if not self.player[y][x] or self.player[y][x] == self.current_player:
                    moves.append((x,y))
        return moves

    def boxes_for(self, player):
        boxes = 0
        for y, row in enumerate(self.board):
            for x, field in enumerate(row):
                if self.player[y][x] == player:
                    boxes+=field
        return boxes

    def value_at(self, x, y):
        return self.board[y][x]

    def player_at(self, x, y):
        return self.player[y][x]

    def _in_board(self, x, y):
        return 0<=x<len(self.board[0]) and 0<=y<len(self.board)

    def winning_player(self):
        if all(x==1 for row in self.player for x in row ):
            return 1
        if all(x==2 for row in self.player for x in row ):
            return 2
        else:
            return 0

    def _throw_over(self, x, y):
        self.board[y][x] -= 4

        if self._in_board(x, y+1):
            self.board[y+1][x] += 1
            self.player[y+1][x] = self.current_player

        if self._in_board(x, y-1):
            self.board[y-1][x] += 1
            self.player[y-1][x] = self.current_player

        if self._in_board(x+1, y):
            self.board[y][x+1] += 1
            self.player[y][x+1] = self.current_player

        if self._in_board(x-1, y):
            self.board[y][x-1] += 1
            self.player[y][x-1] = self.current_player

    def _fields_to_throw(self):
        fields = []
        for y, row in enumerate(self.board):
            for x, field in enumerate(row):
                if field >= 5:
                    fields.append((x,y))
        return fields

    def undo(self):
        current_player, board, player = self.history.pop()
        self.current_player = current_player
        self.board = board
        self.player = player

    def move(self, x, y):
        if self.player[y][x] and not self.player[y][x] == self.current_player:
            raise Exception("Cannot move ontop of other player")

        self.history.append((
            self.current_player,
            deepcopy(self.board),
            deepcopy(self.player)
        ))

        self.board[y][x] += 1
        self.player[y][x] = self.current_player

        if self.board[y][x] == 5:
            self._throw_over(x, y)
            fields = self._fields_to_throw()
            while fields:
                for field in fields:
                    self._throw_over(*field)
                fields = self._fields_to_throw()

        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1

    def print(self):
        print(f"Current board at move {len(self.history)} for player {self.current_player}:")
        for y, row in enumerate(self.board):
            for x, field in enumerate(row):
                player = self.player[y][x]

                if player == 1:
                    color = Board.PLAYER1COLOR
                elif player == 2:
                    color = Board.PLAYER2COLOR
                else:
                    color = Board.ENDC

                print(color+str(field)+Board.ENDC, end=' ')
            print()
        print()

def testMove():
    board = Board()
    assert board.value_at(1, 1) == 0
    assert board.player_at(1, 1) == 0
    assert board.value_at(1, 2) == 0
    assert board.player_at(1, 2) == 0
    board.move(1, 1)
    board.move(1, 2)
    assert board.value_at(1, 1) == 1
    assert board.player_at(1, 1) == 1
    assert board.value_at(1, 2) == 1
    assert board.player_at(1, 2) == 2

def testMoveOntopOfOtherPlayer():
    board = Board()
    board.move(1, 1)

    failed = False
    try:
        board.move(1, 1)
    except:
        failed = True

    assert failed

def testThrowOverMove():
    board  = [
            [4 , 4, 0],
            [0 , 0, 0],
            [4 , 0, 4]
    ]
    player  = [
            [1 , 2, 0],
            [0 , 0, 0],
            [2 , 0, 1]
    ]
    board = Board.from_custom_board(board, player, 1)
    board.move(0, 0)

    assert board.value_at(0, 0) == 2
    assert board.player_at(0, 0) == 1
    assert board.value_at(1, 0) == 1
    assert board.player_at(0, 0) == 1
    assert board.value_at(0, 1) == 1
    assert board.player_at(0, 0) == 1

    assert board.value_at(2, 0) == 1
    assert board.player_at(2, 0) == 1
    assert board.value_at(0, 2) == 4
    assert board.player_at(0, 2) == 2

def testBigFallover():
    board  = [
            [4 , 4, 0],
            [4 , 3, 0],
            [0 , 4, 3]
    ]
    player  = [
            [1 , 1, 0],
            [1 , 2, 0],
            [0 , 2, 2]
    ]
    board = Board.from_custom_board(board, player, 1)
    board.move(0, 0)

    assert board.value_at(2, 2) == 4
    assert board.player_at(2, 2) == 1
    assert board.winning_player() == 1

def testBoxesFor():
    board  = [
            [4 , 4, 0],
            [4 , 3, 0],
            [0 , 4, 3]
    ]
    player  = [
            [1 , 1, 0],
            [1 , 2, 0],
            [0 , 2, 2]
    ]
    board = Board.from_custom_board(board, player, 1)

    assert board.boxes_for(board.current_player) == 12
    assert board.boxes_for(board.other_player) == 10

def run_game(board):
    while not board.winning_player():
        board.print()
        move = input('Please input a move! e.g.: 0,0\n>>')
        if move == 'exit':
            break

        try:
            x, y = move.split(',')
            board.move(int(x), int(y))
        except Exception as e:
            print(e)

def perft(board, AI):
    try:
        from time import time
        print("Starting performance test")
        board.print()
        print("{max_depth} - {move} - {score} - {time} - {nodes_searched} - {pv}")
        for max_depth in range(10):
            ai = AI(max_depth)
            t0 = time()
            move, score = ai.get_best_move(board)
            print(f"{max_depth} - {move} - {score} - {round(time()-t0, 4)} - {ai.stats['moves_made']} - {ai.principle_variation}")
    except KeyboardInterrupt as e:
        pass

if __name__=='__main__':
    testMove()
    testMoveOntopOfOtherPlayer()
    testThrowOverMove()
    testBigFallover()
    testBoxesFor()

    from minmax import MinMax
    from alphabeta import AlphaBeta
    board  = [
            [4 , 0, 0, 3],
            [0 , 3, 0, 4],
            [0 , 4, 4, 0]
    ]
    player  = [
            [1 , 0, 0, 1],
            [0 , 2, 0, 1],
            [0 , 2, 2, 0]
    ]
    board = Board.from_custom_board(board, player, 1)
    perft(board, AlphaBeta)
    perft(board, MinMax)

    #ai = MinMax(2)
    #board.print()
    #print(ai.get_best_move(board), ai.principle_variation)
    #board.print()
