# StackIt Game

This repository will contain the implementation and tutorials for a game called StackIt.
In the tutorial I want to show how to approach implementing a board game with the goal of developing an Ai for it.
The rough outline of the tutorials could be:

* Implement the basic game functionality
* Implement a simple min/max | alpha/beta search Ai
* Implement a MonteCarlo search tree
* Figure out how AlphaGo works and implement a simple version of that on top

I am fairly sure that this will be a *long* way and we will probably rework the board implementation a few times over.

## Running it

```bash
python3 server.py                  # play in a browser: http://localhost:9999
python3 arena.py --help            # pit the engines against each other
python3 bench_search.py --size 5 --time 3   # measure AlphaBeta search depth
```

The web UI offers three ways to play:

* **You against an engine.** It also shows what that engine is thinking: the
  line it expects (principal variation), the moves it weighed, and — for
  AlphaZero — the trained network's own read of the position as a heat map
  over the board.
* **One screen.** Two to five people taking turns on the same device.
* **Online with friends.** Pick two to five seats and you get a four-letter
  code. Send it round; each friend types it into "Join a friend" (or opens
  `/?join=CODE`) and plays from their own browser. There is no lobby and no
  account: whoever has the code is in, the game starts as soon as the last
  seat fills, and the host can start early with whoever turned up. Each
  browser polls once a second for the moves it has not drawn yet, so it
  animates the real cascade rather than jumping to the new position. An
  online game has no undo — there is nobody to ask.

With three or more players a seat is knocked out once every cell belongs to
somebody else: there is nowhere left to drop a block. Last one standing wins.
Only humans can play that game — all three engines search a two-player tree.
Humans-only games may also use a bigger board (up to 12x12) than an engine is
allowed to search (8x8).

AlphaZero only appears as an opponent once a network exists for that board size
(`checkpoints/best.pt`). See [alphazero/README.md](alphazero/README.md) for
training.

## Deploying it

To put the server on a machine of your own, see [DEPLOY.md](DEPLOY.md). Short
version: Python 3.10+, `pip install -r requirements.txt` (with the CPU-only
PyTorch index), gunicorn behind your reverse proxy, and a copy of
`checkpoints/best.pt` — which is gitignored, so it will not arrive with the
code. It fits a 1 GB, single-core box.

One rule worth repeating here: run **one** gunicorn worker with several
threads. Games are held in the process's memory, so a second worker process
loses half of them.

## Where the code lives

| file | role |
|---|---|
| `board.py` | the game itself: moves, chain reactions, Zobrist hashing |
| `alphabeta.py` | alpha-beta search engine (PVS, killers, transposition table) |
| `mcts.py` | plain Monte-Carlo tree search engine |
| `alphazero/` | self-play trainer + neural-network engine |
| `server.py` | JSON API + web UI |
| `arena.py` | engine-vs-engine tournaments |
| `bench_search.py` | search-depth benchmark |

## The Game

The idea for the game of StackIt is fairly simple.
Players take turns and stack blocks on a board.
A player can either add a block on a field that already contains his own blocks or put a block on an empty field,
coloring the field in his color, making it his own.
If a field contains more than four blocks, they fall over and spill to top, bottom, left and right, coloring those fields in his.
If any of the fields contains more than five blocks after that it will also immediately spill and create chain reaction.

The default board will be 5x5 and 
it is for two players but could easily be played with more.

Example:

```
Notation:
##  - Empty field
XC  - X blocks of color C e.g. 4R means 4 red blocks

On a 3x3 board it could look like this:
  1 2 3
1 ######
2 ##4R##
3 ######
```

if the red player now puts another box on field (2,2) then it will spill over producing this board:

```
  1 2 3
1 ##1R##
2 1R1R1R
3 ##1R##
```


Lets assume that there was a stack of four blue blocks next to it like:

```
  1 2 3
1 ######
2 ##4R4B
3 ######
```

Upon collapsing the four red blocks it would spill onto the blue ones, coloring them red and immediately collapse this aswell

```
  1 2 3
1 ##1R1R
2 1R1R1R
3 ##1R1R
```

## Ideas

- Currently blocks falling of the board would disappear but it could be considered that they just stay on the inital field speeding up the next collapse
- In order to speed up the initial game it could be a good idea to immediately place 3 blocks on empty fields, this has implications for the early midgame though so needs to be tested
