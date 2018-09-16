# First MinMax AI algorithm

MinMax is an algorithm that successively tries to play all possible moves 
trying to maximize the score in one round and minimizing the score in the second.

The idea is that the current player will want to play the move,
that gives the maximum return while the opponent will play the move which minimizes the score for the current player.

It in some of its refined versions is and used to be very popular for chess and should be a good match for our game of StackIt since it is simple and easy to implement.
We will implement it in its simple form and then add a couple of the improvements to probably get it to the point where it is 

a) unbeatable by me
b) it finds some interesting patterns on how to beat me
c) maybe it already gives some interesting self play games

If you want to know more about the particular algorithm you can google for some more tutorials or graphics that explain it better than I possibly could.

As for the implementation we will make a class that takes the depth to which we want to search as input.
Later we will replace the depth parameter with a time that the algorithm is allowed to search.

We will have a best_move function which will do some convenience things and return the score of the core part which is the
maximize function with this defintion:

```
def maximize(board, depth) -> best_move, score
```

Maxmimize will given a depth find the best move and return the score of that move.
If the depth is zero e it should just return a score (leaf node in the search tree e.g. depth=0 or someone won with the last move)
Else it should get all possible moves and try them one by one trying to find which of those moves is the best.
The way this works is that it will do a move on the board and call itself from the perspective of the other player trying to then minimize the score.

Since the "minimizing" is doing the same thing as the maximizing function except that the higher the score returned from the perspective of the other player the worse it is for the current player.
So we call maximize and negate the score trying to find the "lowest" score for the opponent hence maximizing our own score

So this is the core of the algorithm:

```python
    for move in board.possible_moves():
            board.move(*move)
            _, score = self._maxmimize(board, depth-1)
            score*=-1
            board.undo()
            if score > best_score:
                best_score = score
                best_move = move

```

Another tricky part that we need to come up with is, if we end in a leaf node (searched all the way to depth n) and its not a winning position for either player,
how good do we think the position is for the player that made the last move.

In chess a simple way of doing this is counting the pieces for both sides weighted by how important they are and subtracting them.

We will use a similar approach initially and count the number of boxes belonging to either player and returning the difference as the score,
the more boxes I have, the better it is for me.

```python
    if depth == 0:
        return None, board.boxes_for(board.current_player) - board.boxes_for(board.other_player)
    if  board.winning_player():
        return None, -1000000
```

Of course in case someone won, we return a highest possible score
 
The full code:

```python
class MinMax:
    def __init__(self, depth):
        self.depth = depth

    def get_best_move(self, board):
        res =  self._maxmimize(board, self.depth)
        return res

    def _maxmimize(self, board, depth):
        if depth == 0:
            return None, board.boxes_for(board.current_player) - board.boxes_for(board.other_player)
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

```


