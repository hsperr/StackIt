import pytest
from utils import StackItException
from board import Board
from minmax import MinMax
from mcts import MonteCarloTreeSearch
from alphabeta import rotate_move, flip_move

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


def test_undo_with_no_history_raises():
    board = Board()
    with pytest.raises(StackItException):
        board.undo()


def test_undo_after_moves_then_empty_raises():
    board = Board()
    board.move(0, 0)
    board.undo()
    assert board.value_at(0, 0) == 0
    with pytest.raises(StackItException):
        board.undo()


def test_to_string_from_string_roundtrip():
    board = Board(3, 3)
    board.move(1, 1)
    board.move(0, 0)
    board.move(2, 2)

    restored = Board.from_string(board.to_string())

    assert restored.board == board.board
    assert restored.player == board.player
    assert restored.current_player == board.current_player


def test_minmax_get_best_move_does_not_crash_near_end_of_game():
    # Board is nearly fully owned by player 1, with only one free cell
    # left; this exercises MinMax._maxmimize's move loop and return path
    # (previously `principle_variation` could be referenced before
    # assignment there).
    board_arr = [
        [1, 1],
        [1, 0],
    ]
    player_arr = [
        [1, 1],
        [1, 0],
    ]
    board = Board.from_custom_board(board_arr, player_arr, 1)
    original_board = Board.from_custom_board(board_arr, player_arr, 1)

    move, score = MinMax(max_depth=2).get_best_move(board)

    assert move in original_board.possible_moves()
    assert isinstance(score, int)
    # get_best_move leaves the board as it found it (moves/undos in pairs).
    assert board.board == original_board.board
    assert board.player == original_board.player


def test_mcts_get_best_move_returns_tuple_with_no_moves():
    board_arr = [[1, 1], [1, 1]]
    player_arr = [[2, 2], [2, 2]]
    board = Board.from_custom_board(board_arr, player_arr, 1)

    result = MonteCarloTreeSearch().get_best_move(board, thinking_time=0)

    assert result == (None, None)
    move, score = result  # must be unpackable like the main-path return


def test_mcts_get_best_move_returns_tuple_with_single_move():
    board_arr = [[1, 0], [0, 0]]
    player_arr = [[1, 2], [2, 2]]
    board = Board.from_custom_board(board_arr, player_arr, 1)

    move, score = MonteCarloTreeSearch().get_best_move(board, thinking_time=0)

    assert move == (0, 0)
    assert score == 0


def test_rotate_move_matches_actual_board_rotation():
    board = Board(4, 4)
    for y in range(board.size_y):
        for x in range(board.size_x):
            board.board[y][x] = y * board.size_x + x
    rotated = board.rotate()

    for y in range(board.size_y):
        for x in range(board.size_x):
            value = board.board[y][x]
            new_x, new_y = rotate_move((x, y), rotated.size_x, rotated.size_y)
            assert rotated.board[new_y][new_x] == value


def test_flip_move_matches_actual_board_flip():
    board = Board(4, 3)
    for y in range(board.size_y):
        for x in range(board.size_x):
            board.board[y][x] = y * board.size_x + x
    flipped = board.flip()

    for y in range(board.size_y):
        for x in range(board.size_x):
            value = board.board[y][x]
            new_x, new_y = flip_move((x, y), flipped.size_x, flipped.size_y)
            assert flipped.board[new_y][new_x] == value

