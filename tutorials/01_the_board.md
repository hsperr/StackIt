# Setup and Board

In order to run this repo you will need:

* Python 3.6+ (I use miniconda and install the packages I need on the go)


## The Board

In order to implement the board we start out by making a simple class that is able to print itself
First here is the full first draft of the class. Later on we may want to switch it to use different data structures or modules
to improve performance but as a first MVP this works.

### The board itself
The idea is that we store the "boxes" and the color of the box on that field in separate arrays.
For now we support two players and players set one box at a time. 
Later on I may want to play with the idea that you can place 3 boxes on empty fields to speed up the inital game.

### Move function
In our move function we check if our move is legal and make towers fall over if they are too big.
Once towers fell over we need to check if new towers are ready to fall and do this until all towers fell over.
We decide that a box that falls off the grid will be removed there could be other fun ways:

1. Remove the box falling off the field
2. Leave the box on the current field
3. Let the box "bounce" into the opposite direction of the wall
4. Let the box randomly fall into one possible direction
5. Let the box wrap around and enter the board from the other side

### Undo
We will need a proper undo, especially for the AI later that will modify the board and play moves on it to find the best continuation
For now undo will be as simple as making a copy of the state of the board and on undo overriding the current state.

### Helpers

We have a couple of helpers that let us get players and number of boxes for fields, get the total number of boxes for a given player or
get all possible moves for the current player or return the winning player.

## The Code

```Python
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

    def num_boxes_for(self, player):
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
```




