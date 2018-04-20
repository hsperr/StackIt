# StackIt Game

This repository will contain the implementation and tutorials for a game called StackIt.
In the tutorial I want to show how to approach implementing a board game with the goal of developing an Ai for it.
The rough outline of the tutorials could be:

* Implement the basic game functionality
* Implement a simple min/max | alpha/beta search Ai
* Implement a MonteCarlo search tree
* Figure out how AlphaGo works and implement a simple version of that on top

I am fairly sure that this will be a *long* way and we will probably rework the board implementation a few times over.

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
