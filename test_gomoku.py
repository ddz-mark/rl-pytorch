import pytest

from gomoku import GomokuGame, parse_move


def test_horizontal_win_detection() -> None:
    game = GomokuGame(size=10, win_length=5)
    for col in range(5):
        game.board[3][col] = "X"
    assert game.check_win(3, 2) is True


def test_diagonal_win_after_moves() -> None:
    game = GomokuGame(size=10, win_length=5)
    moves = [
        (0, 0), (0, 1),
        (1, 1), (0, 2),
        (2, 2), (0, 3),
        (3, 3), (0, 4),
        (4, 4),
    ]
    for row, col in moves:
        game.place_stone(row, col)
    assert game.winner == "X"


def test_occupied_cell_rejected() -> None:
    game = GomokuGame(size=8, win_length=5)
    game.place_stone(0, 0)
    with pytest.raises(ValueError, match="occupied"):
        game.place_stone(0, 0)


def test_parse_move_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="two numbers"):
        parse_move("1", 15)
    with pytest.raises(ValueError, match="integers"):
        parse_move("1 x", 15)
    with pytest.raises(ValueError, match="between 1 and 15"):
        parse_move("16 1", 15)


def test_draw_condition() -> None:
    game = GomokuGame(size=3, win_length=3)
    sequence = [
        (0, 0), (0, 1), (0, 2),
        (1, 1), (1, 0), (1, 2),
        (2, 1), (2, 0), (2, 2),
    ]
    for row, col in sequence:
        game.place_stone(row, col)
    assert game.is_draw() is True
    assert game.winner is None
