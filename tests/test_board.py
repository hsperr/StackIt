import pytest
from ..utils import StackItException
from ..board import Board

def test_move():
    board = Board()
    assert board.value_at(1, 1) == 0
    assert board.player_at(1, 1) == 0
    assert board.value_at(1, 2) == 0
    assert board.player_at(1, 2) == 0

    board.move(1, 1)
    board.move(1, 2)

    assert board.value_at(1, 1) == board.INITAL_BOX_INCREASE
    assert board.player_at(1, 1) == 1
    assert board.value_at(1, 2) == board.INITAL_BOX_INCREASE
    assert board.player_at(1, 2) == 2

def test_move_ontop_of_other_player():
    board = Board()
    board.move(1, 1)

    with pytest.raises(StackItException):
        board.move(1, 1)

def test_throw_over_move():
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

def test_big_fallover():
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

def test_boxes_for():
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


