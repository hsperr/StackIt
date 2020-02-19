from copy import deepcopy
from utils import StackItException
import math
import time


class Board:
    PLAYER1COLOR = '\033[92m'
    PLAYER2COLOR = '\033[91m'
    ENDC = '\033[0m'

    INITAL_BOX_INCREASE = 3

    @classmethod
    def from_string(cls, board_string):
        board_string = board_string.replace(" ", '')
        board_string = board_string.replace("\n", '')

        instance = cls()
        instance.current_player = int(board_string[0])
        board_string = board_string[1:]

        def parse_row(row):
            board, player = [], []
            for i in range(0, len(row), 2):
                board.append(int(row[i]))
                player.append(int(row[i + 1]))

            return board, player

        field_size = int(math.sqrt(len(board_string) / 2)) * 2
        board, player = [], []
        for i in range(0, len(board_string), field_size):
            temp_board, temp_player = parse_row(board_string[i:i + field_size])
            board.append(temp_board)
            player.append(temp_player)

        instance.board = board
        instance.player = player
        return instance

    def to_string(self):
        return ''.join([str(x) for x in [self.current_player] + self.board + self.player])

    def hash(self):
        return hash(self.to_string())

    def copy(self):
        return Board.from_custom_board(self.board, self.player, self.current_player)

    @classmethod
    def from_custom_board(cls, board, player, current_player=1):
        import copy
        instance = cls()
        instance.board = copy.deepcopy(board)
        instance.player = copy.deepcopy(player)
        instance.current_player = current_player
        return instance

    def __init__(self, size_x=5, size_y=5):
        self.board = [[0 for _ in range(size_x)] for _ in range(size_y)]
        self.player = [[0 for _ in range(size_x)] for _ in range(size_y)]

        self.current_player = 1

        self.history = []

    @property
    def size_x(self):
        return len(self.board[0])

    @property
    def size_y(self):
        return len(self.board)

    @property
    def other_player(self):
        if self.current_player == 1:
            return 2
        else:
            return 1

    def flip(self):
        b = Board()
        b.current_player = self.current_player
        b.board = self.board[::-1]
        b.player = self.player[::-1]
        return b

    def rotate(self):
        b = Board()
        b.current_player = self.current_player
        b.board = [list(x) for x in zip(*self.board[::-1])]
        b.player = [list(x) for x in zip(*self.player[::-1])]
        return b

    def possible_attack_moves(self):
        moves = []
        for y, row in enumerate(self.board):
            for x, field in enumerate(row):
                if self.player[y][x] == self.current_player and field == 4:
                    moves.append((x, y))
        return moves

    def possible_moves(self):
        moves = []
        for y, row in enumerate(self.board):
            for x, field in enumerate(row):
                if not self.player[y][x] or self.player[y][x] == self.current_player:
                    moves.append((field if field else 3 + (abs(x-self.size_x//2) + abs(y-self.size_y//2))/10, x, y))
        moves.sort()
        return [(x[-2], x[-1]) for x in moves]

    def boxes_for(self, player):
        boxes = 0
        for y, row in enumerate(self.board):
            for x, field in enumerate(row):
                if self.player[y][x] == player:
                    boxes += field
        return boxes

    def value_at(self, x, y):
        return self.board[y][x]

    def player_at(self, x, y):
        return self.player[y][x]

    def _in_board(self, x, y):
        return 0 <= x < len(self.board[0]) and 0 <= y < len(self.board)

    def winning_player(self):
        if all(x == 1 for row in self.player for x in row):
            return 1
        if all(x == 2 for row in self.player for x in row):
            return 2
        else:
            return 0

    def _throw_over(self, x, y):
        self.board[y][x] -= 4

        if self._in_board(x, y + 1):
            self.board[y + 1][x] += 1
            self.player[y + 1][x] = self.current_player

        if self._in_board(x, y - 1):
            self.board[y - 1][x] += 1
            self.player[y - 1][x] = self.current_player

        if self._in_board(x + 1, y):
            self.board[y][x + 1] += 1
            self.player[y][x + 1] = self.current_player

        if self._in_board(x - 1, y):
            self.board[y][x - 1] += 1
            self.player[y][x - 1] = self.current_player

    def _fields_to_throw(self):
        fields = []
        for y, row in enumerate(self.board):
            for x, field in enumerate(row):
                if field >= 5:
                    fields.append((x, y))
        return fields

    def undo(self):
        current_player, board, player = self.history.pop()
        self.current_player = current_player
        self.board = board
        self.player = player

    def move(self, x, y, display=False):
        if self.player[y][x] and not self.player[y][x] == self.current_player:
            raise StackItException(f"Cannot move ontop of other player (current_player={self.current_player}, x={x}, y={y}, field={self.player[y][x]})")

        self.history.append((
            self.current_player,
            deepcopy(self.board),
            deepcopy(self.player)
        ))

        if self.board[y][x] == 0:
            self.board[y][x] += Board.INITAL_BOX_INCREASE
        else:
            self.board[y][x] += 1

        self.player[y][x] = self.current_player

        if self.board[y][x] == 5:
            self._throw_over(x, y)
            if display:
                self.print()
                time.sleep(1)
            fields = self._fields_to_throw()
            while fields:
                for field in fields:
                    self._throw_over(*field)
                    if display:
                        self.print()
                        time.sleep(1)
                fields = self._fields_to_throw()

        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1

    def _player_color(self, player):
        if player == 1:
            return Board.PLAYER1COLOR
        elif player == 2:
            return Board.PLAYER2COLOR

        return Board.ENDC

    def print(self):
        current_color = self._player_color(self.current_player)
        spacing = "  "
        print(f"Current board at move {len(self.history)} for player {current_color + str(self.current_player) + Board.ENDC}:")
        print()
        print(spacing, ' ', ' '.join([str(x) for x in range(self.size_x)]))
        print(spacing, ' ', ' '.join(['-' for x in range(self.size_x)]))
        for y, row in enumerate(self.board):
            print(spacing + str(y) + '|', end=' ')
            for x, field in enumerate(row):
                color = self._player_color(self.player[y][x])
                print(color + str(field) + Board.ENDC, end=' ')
            print()
        print()


if __name__ == '__main__':
    board_string = """
            2
            41 31 11
            11 42 31
            42 42 42
        """
    board = Board.from_string(board_string)
    board.print()
