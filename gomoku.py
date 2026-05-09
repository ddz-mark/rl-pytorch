#!/usr/bin/env python3
"""Terminal Gomoku mini game."""

from __future__ import annotations

BOARD_SIZE = 15
EMPTY = "."
PLAYERS = ("X", "O")


def create_board(size: int = BOARD_SIZE) -> list[list[str]]:
    return [[EMPTY for _ in range(size)] for _ in range(size)]


def render_board(board: list[list[str]]) -> None:
    size = len(board)
    header = "   " + " ".join(f"{i:2d}" for i in range(size))
    print(header)
    for r, row in enumerate(board):
        print(f"{r:2d} " + " ".join(f" {cell}" for cell in row))


def parse_move(raw: str, size: int) -> tuple[int, int] | None:
    text = raw.strip().lower()
    if text in {"q", "quit", "exit"}:
        return None

    parts = raw.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError("请输入两个坐标，例如: 7 7")

    row, col = int(parts[0]), int(parts[1])
    if not (0 <= row < size and 0 <= col < size):
        raise ValueError(f"坐标必须在 0 到 {size - 1} 之间")

    return row, col


def has_five(board: list[list[str]], row: int, col: int, player: str) -> bool:
    size = len(board)
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))

    for dr, dc in directions:
        count = 1

        rr, cc = row + dr, col + dc
        while 0 <= rr < size and 0 <= cc < size and board[rr][cc] == player:
            count += 1
            rr += dr
            cc += dc

        rr, cc = row - dr, col - dc
        while 0 <= rr < size and 0 <= cc < size and board[rr][cc] == player:
            count += 1
            rr -= dr
            cc -= dc

        if count >= 5:
            return True

    return False


def is_board_full(board: list[list[str]]) -> bool:
    return all(cell != EMPTY for row in board for cell in row)


def run_game() -> None:
    board = create_board()
    current = 0

    print("欢迎来到五子棋小游戏!")
    print("输入格式: 行 列 (例如: 7 7)，输入 q 退出。")

    while True:
        render_board(board)
        player = PLAYERS[current]

        try:
            raw = input(f"玩家 {player} 落子> ")
            move = parse_move(raw, len(board))
            if move is None:
                print("游戏已退出。")
                return

            row, col = move
            if board[row][col] != EMPTY:
                print("该位置已有棋子，请重试。")
                continue

            board[row][col] = player

            if has_five(board, row, col, player):
                render_board(board)
                print(f"玩家 {player} 获胜! 🎉")
                return

            if is_board_full(board):
                render_board(board)
                print("棋盘已满，平局。")
                return

            current = 1 - current

        except ValueError as err:
            print(f"输入错误: {err}")
        except KeyboardInterrupt:
            print("\n游戏已中断。")
            return


if __name__ == "__main__":
    run_game()
