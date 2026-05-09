#!/usr/bin/env python3
"""Terminal Gomoku (Five in a Row) mini game.

Run:
    python gomoku.py
"""

from __future__ import annotations

BOARD_SIZE = 15
EMPTY = "."
PLAYER_X = "X"
PLAYER_O = "O"


def create_board(size: int) -> list[list[str]]:
    return [[EMPTY for _ in range(size)] for _ in range(size)]


def print_board(board: list[list[str]]) -> None:
    size = len(board)
    header = "   " + " ".join(f"{c:2d}" for c in range(size))
    print("\n" + header)
    for r in range(size):
        row = " ".join(f" {cell}" for cell in board[r])
        print(f"{r:2d} {row}")
    print()


def parse_move(raw: str, size: int) -> tuple[int, int] | None:
    raw = raw.strip().replace(",", " ")
    parts = [p for p in raw.split() if p]
    if len(parts) != 2:
        return None
    try:
        r, c = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if 0 <= r < size and 0 <= c < size:
        return r, c
    return None


def is_winner(board: list[list[str]], row: int, col: int, player: str) -> bool:
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    size = len(board)

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


def is_full(board: list[list[str]]) -> bool:
    return all(cell != EMPTY for row in board for cell in row)


def play() -> None:
    board = create_board(BOARD_SIZE)
    current = PLAYER_X

    print("欢迎来到五子棋小游戏！")
    print("输入格式: 行 列 (例如: 7 7)，输入 q 退出。")

    while True:
        print_board(board)
        move_raw = input(f"玩家 {current} 落子 > ").strip()

        if move_raw.lower() in {"q", "quit", "exit"}:
            print("游戏已退出。")
            return

        move = parse_move(move_raw, BOARD_SIZE)
        if move is None:
            print("输入无效，请输入两个合法整数坐标，例如: 7 7")
            continue

        row, col = move
        if board[row][col] != EMPTY:
            print("该位置已有棋子，请重新选择。")
            continue

        board[row][col] = current

        if is_winner(board, row, col, current):
            print_board(board)
            print(f"玩家 {current} 获胜！")
            return

        if is_full(board):
            print_board(board)
            print("平局，棋盘已满。")
            return

        current = PLAYER_O if current == PLAYER_X else PLAYER_X


def main() -> None:
    play()


if __name__ == "__main__":
    main()
