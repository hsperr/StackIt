from utils import StackItException
import math
import time
import random


class Board:
    PLAYER1COLOR = '\033[92m'
    PLAYER2COLOR = '\033[91m'
    ENDC = '\033[0m'

    INITAL_BOX_INCREASE = 3

    # --- Zobrist hashing -------------------------------------------------
    # Incremental 64-bit hash of (cell values, owners, side-to-move). Kept up
    # to date inside move()/undo() so the search never has to serialize the
    # board to a string for its transposition-table key (that to_string() call
    # used to dominate search time). Tables are built once per board size and
    # shared across all boards of that size.
    _ZOBRIST = {}          # (size_x, size_y) -> (piece_table, side_key)
    _ZVMAX = 16            # max cell value covered (stable 0-4, transient <=8)

    @classmethod
    def _zobrist_tables(cls, size_x, size_y):
        tables = cls._ZOBRIST.get((size_x, size_y))
        if tables is None:
            # Fixed seed -> reproducible keys across runs/processes (matters for
            # the parallel self-play workers that share a TT-less protocol but
            # still benefit from deterministic behaviour in tests).
            rng = random.Random(0x57ACC17 ^ (size_x << 8) ^ size_y)
            ncells = size_x * size_y
            # piece[pos][owner][value]. Empty cells (owner 0, value 0) are
            # forced to 0 so they contribute nothing to the key.
            piece = [[[rng.getrandbits(64) for _ in range(cls._ZVMAX + 1)]
                      for _ in range(3)] for _ in range(ncells)]
            for pos in range(ncells):
                piece[pos][0][0] = 0
            tables = (piece, rng.getrandbits(64))
            cls._ZOBRIST[(size_x, size_y)] = tables
        return tables

    def _init_zobrist(self):
        """Bind this board to its size's Zobrist tables and compute the key
        from scratch. Call once, after board/player/current_player are final."""
        self._sx = len(self.board[0])
        self._piece, self._side = Board._zobrist_tables(self._sx, len(self.board))
        self.zkey = self._compute_zkey()

    def _compute_zkey(self):
        piece, z, pos = self._piece, 0, 0
        for brow, prow in zip(self.board, self.player):
            for v, p in zip(brow, prow):
                z ^= piece[pos][p][v]
                pos += 1
        if self.current_player == 2:
            z ^= self._side
        return z

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
        instance._init_zobrist()
        return instance

    def to_string(self):
        cells = []
        for y, row in enumerate(self.board):
            for x, field in enumerate(row):
                cells.append(str(field))
                cells.append(str(self.player[y][x]))
        return str(self.current_player) + ''.join(cells)

    def hash(self):
        return self.zkey

    def copy(self):
        return Board.from_custom_board(self.board, self.player, self.current_player)

    @classmethod
    def from_custom_board(cls, board, player, current_player=1):
        instance = cls()
        # Row-slice copy: fully independent (ints are immutable) and much
        # cheaper than deepcopy for this list-of-lists-of-ints shape.
        instance.board = [row[:] for row in board]
        instance.player = [row[:] for row in player]
        instance.current_player = current_player
        instance._init_zobrist()
        return instance

    def __init__(self, size_x=5, size_y=5):
        self.board = [[0 for _ in range(size_x)] for _ in range(size_y)]
        self.player = [[0 for _ in range(size_x)] for _ in range(size_y)]

        self.current_player = 1

        self.history = []
        self._init_zobrist()

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
        # Bypass __init__ so we don't allocate two zero grids only to discard
        # them; these symmetry boards are used purely for hashing.
        b = Board.__new__(Board)
        b.current_player = self.current_player
        b.board = self.board[::-1]
        b.player = self.player[::-1]
        b.history = []
        b._init_zobrist()
        return b

    def rotate(self):
        b = Board.__new__(Board)
        b.current_player = self.current_player
        b.board = [list(x) for x in zip(*self.board[::-1])]
        b.player = [list(x) for x in zip(*self.player[::-1])]
        b.history = []
        b._init_zobrist()
        return b

    def possible_attack_moves(self):
        moves = []
        cur = self.current_player
        for y, (brow, prow) in enumerate(zip(self.board, self.player)):
            for x, field in enumerate(brow):
                if prow[x] == cur and field == 4:
                    moves.append((x, y))
        return moves

    def possible_moves(self):
        # Hoist the size/current_player lookups out of the inner loop and walk
        # the board/player grids together to avoid repeated double-indexing and
        # @property (len) calls. Output order is byte-for-byte identical.
        cur = self.current_player
        hx = len(self.board[0]) // 2
        hy = len(self.board) // 2
        moves = []
        for y, (brow, prow) in enumerate(zip(self.board, self.player)):
            for x, (field, p) in enumerate(zip(brow, prow)):
                if not p or p == cur:
                    moves.append((field if field else 3 + (abs(x - hx) + abs(y - hy)) / 10, x, y))
        moves.sort()
        return [(m[1], m[2]) for m in moves]

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
        # Single pass instead of two full scans: track whether the grid is
        # all-1s / all-2s and bail out as soon as neither is possible.
        p1 = p2 = True
        for row in self.player:
            for v in row:
                if v != 1:
                    p1 = False
                if v != 2:
                    p2 = False
                if not p1 and not p2:
                    return 0
        return 1 if p1 else (2 if p2 else 0)

    def _throw_over(self, x, y):
        # Every cell mutation here also folds the change into self.zkey (XOR out
        # the old (owner, value), XOR in the new). Bounds are inlined instead of
        # calling _in_board so the cascade stays cheap.
        b, p, piece, sx = self.board, self.player, self._piece, self._sx
        cur = self.current_player
        sy = len(b)
        pos = y * sx + x

        self.zkey ^= piece[pos][p[y][x]][b[y][x]]
        b[y][x] -= 4
        self.zkey ^= piece[pos][p[y][x]][b[y][x]]

        if y + 1 < sy:
            self.zkey ^= piece[pos + sx][p[y + 1][x]][b[y + 1][x]]
            b[y + 1][x] += 1
            p[y + 1][x] = cur
            self.zkey ^= piece[pos + sx][cur][b[y + 1][x]]

        if y - 1 >= 0:
            self.zkey ^= piece[pos - sx][p[y - 1][x]][b[y - 1][x]]
            b[y - 1][x] += 1
            p[y - 1][x] = cur
            self.zkey ^= piece[pos - sx][cur][b[y - 1][x]]

        if x + 1 < sx:
            self.zkey ^= piece[pos + 1][p[y][x + 1]][b[y][x + 1]]
            b[y][x + 1] += 1
            p[y][x + 1] = cur
            self.zkey ^= piece[pos + 1][cur][b[y][x + 1]]

        if x - 1 >= 0:
            self.zkey ^= piece[pos - 1][p[y][x - 1]][b[y][x - 1]]
            b[y][x - 1] += 1
            p[y][x - 1] = cur
            self.zkey ^= piece[pos - 1][cur][b[y][x - 1]]

    def _fields_to_throw(self):
        fields = []
        for y, row in enumerate(self.board):
            for x, field in enumerate(row):
                if field >= 5:
                    fields.append((x, y))
        return fields

    def undo(self):
        if not self.history:
            raise StackItException("Cannot undo, no moves have been made")
        current_player, board, player, zkey = self.history.pop()
        self.current_player = current_player
        self.board = board
        self.player = player
        self.zkey = zkey

    def move(self, x, y, display=False):
        if self.player[y][x] and not self.player[y][x] == self.current_player:
            raise StackItException(f"Cannot move ontop of other player (current_player={self.current_player}, x={x}, y={y}, field={self.player[y][x]})")

        # Grids are lists-of-lists-of-ints, so a per-row slice copy produces a
        # fully independent snapshot far faster than copy.deepcopy (which walks
        # the object graph via reflection). This is the hottest path in search.
        self.history.append((
            self.current_player,
            [row[:] for row in self.board],
            [row[:] for row in self.player],
            self.zkey
        ))

        piece = self._piece
        pos = y * self._sx + x
        # XOR out the moved cell's old (owner, value) before mutating it.
        self.zkey ^= piece[pos][self.player[y][x]][self.board[y][x]]
        if self.board[y][x] == 0:
            self.board[y][x] += Board.INITAL_BOX_INCREASE
        else:
            self.board[y][x] += 1

        self.player[y][x] = self.current_player
        self.zkey ^= piece[pos][self.current_player][self.board[y][x]]

        if self.board[y][x] >= 5:
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

        # Flip side to move; toggling the side key XORs it in/out symmetrically.
        self.zkey ^= self._side
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
