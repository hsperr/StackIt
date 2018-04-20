# Implementing the Board

In order to implement the game board we will first define the interface and then write tests and implement against it.

My usual approach is a bit more messy but lets see if we can figure things out on the fly.

Lets think about what the inital basic board needs to support:

* It needs to store some information of the current state of the NxN fields (lets assume 5x5 is the default now)
* It needs to be able to give us information about certain fields 
* It needs to be able to give the valid moves for the current player
* It needs to be able to validate a move
* It needs to be able to make a move and update the state accordingly

In the first implementation we will just use a numpy matrix to store the current board, later on we may want to come up with different solutions.

## Board Interface

```python
class Board:
    def get_value(self, x, y):
        pass

    def size_x(self):
        pass

    def size_y(self):
        pass

    def get_moves(self):
        pass

    def make_move(self):
        pass

```



## ArrayBoard

